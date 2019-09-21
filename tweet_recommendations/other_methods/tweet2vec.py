import pickle as pkl
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Union

import lasagne
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from functional import seq

from tweet_recommendations.other_methods.method import Method
from .tweet2vec.tweet2vec import batch_char as batch, settings_char, t2v


class Tweet2Vec(Method):
    def __init__(
        self,
        epochs: int,
        model_path: str,
        verbose: bool = False,
        last_epoch: Optional[int] = None,
    ):
        self.model_path = Path(model_path)
        self.epochs = epochs
        self.verbose = verbose
        self.max_classes = 999999
        self.last_epoch = last_epoch

    def fit(
        self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params
    ) -> "Method":
        if y is not None:
            self.max_classes = len(np.unique(y))

        # Training data
        Xt = []
        self._print("Building Model...")
        if self.model_path.exists():
            self._print("Loading model params...")
            params = t2v.load_params_shared(
                (self.model_path / "model.npz").as_posix()
            )

            self._print("Loading dictionaries...")
            with open((self.model_path / "dict.pkl").as_posix(), "rb") as f:
                chardict = pkl.load(f)
            with open(
                (self.model_path / "label_dict.pkl").as_posix(), "rb"
            ) as f:
                labeldict = pkl.load(f)
            n_char = len(list(chardict.keys())) + 1
            n_classes = min(len(list(labeldict.keys())) + 1, self.max_classes)

        else:
            # Build dictionaries from training data
            chardict, charcount = batch.build_dictionary(Xt)
            n_char = len(list(chardict.keys())) + 1
            batch.save_dictionary(
                chardict, charcount, (self.model_path / "dict.pkl").as_posix()
            )
            # params
            params = t2v.init_params(n_chars=n_char)

            labeldict, labelcount = batch.build_label_dictionary(y)
            batch.save_dictionary(
                labeldict, labelcount, (self.model_path / "label_dict.pkl")
            )

            n_classes = min(len(list(labeldict.keys())) + 1, self.max_classes)

            # classification params
            params["W_cl"] = theano.shared(
                np.random.normal(
                    loc=0.,
                    scale=settings_char.SCALE,
                    size=(settings_char.WDIM, n_classes),
                ).astype("float32"),
                name="W_cl",
            )
            params["b_cl"] = theano.shared(
                np.zeros((n_classes,)).astype("float32"), name="b_cl"
            )

        # iterators
        train_iter = batch.BatchTweets(
            x,
            y,
            labeldict,
            batch_size=settings_char.N_BATCH,
            max_classes=self.max_classes,
        )

        self._print("Building network...")
        # Tweet variables
        tweet = T.itensor3()
        targets = T.ivector()
        # masks
        t_mask = T.fmatrix()

        # network for prediction
        predictions, net, emb = self._classify(
            tweet, t_mask, params, n_classes, n_char
        )

        # batch loss
        loss = lasagne.objectives.categorical_crossentropy(
            predictions, targets
        )
        cost = T.mean(
            loss
        ) + settings_char.REGULARIZATION * lasagne.regularization.regularize_network_params(
            net, lasagne.regularization.l2
        )
        cost_only = T.mean(loss)
        reg_only = (
            settings_char.REGULARIZATION
            * lasagne.regularization.regularize_network_params(
                net, lasagne.regularization.l2
            )
        )

        # params and updates
        self._print("Computing updates...")
        lr = settings_char.LEARNING_RATE
        mu = settings_char.MOMENTUM
        updates = lasagne.updates.nesterov_momentum(
            cost, lasagne.layers.get_all_params(net), lr, momentum=mu
        )

        # Theano function
        self._print("Compiling theano functions...")
        inps = [tweet, t_mask, targets]
        predict = theano.function([tweet, t_mask], predictions)
        cost_val = theano.function(inps, [cost_only, emb])
        train = theano.function(inps, cost, updates=updates)
        reg_val = theano.function([], reg_only)

        # Training
        self._print("Training...")
        uidx = 0
        maxp = 0.
        try:
            for epoch in range(self.epochs):
                n_samples = 0
                train_cost = 0.
                self._print(("Epoch {}".format(epoch)))
                for xr, y in train_iter:
                    n_samples += len(xr)
                    uidx += 1
                    x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
                    if x is None:
                        self._print(
                            "Minibatch with zero samples under maxlength."
                        )
                        uidx -= 1
                        continue

                    curr_cost = train(x, x_m, y)
                    train_cost += curr_cost * len(xr)

                    if np.isnan(curr_cost) or np.isinf(curr_cost):
                        self._print("Nan detected.")
                        return self

                    if np.mod(uidx, settings_char.DISPF) == 0:
                        self._print(
                            (
                                "Epoch {} Update {} Cost {}".format(
                                    epoch, uidx, curr_cost
                                )
                            )
                        )

                if np.mod(uidx, settings_char.SAVEF) == 0:
                    self._print("Saving...")
                    saveparams = OrderedDict()
                    for kk, vv in params.items():
                        saveparams[kk] = vv.get_value()
                    np.savez(
                        (self.model_path / "model.npz").as_posix(),
                        **saveparams
                    )
                    self._print("Done.")

                    self._print("Testing on Validation set...")
                self._print(
                    (
                        "Epoch {} Training Cost {}  Max Precision {}".format(
                            epoch, train_cost / n_samples, maxp
                        )
                    )
                )
                self._print(("Seen {} samples.".format(n_samples)))

                self._print("Saving...")
                saveparams = OrderedDict()
                for kk, vv in params.items():
                    saveparams[kk] = vv.get_value()
                np.savez(
                    (
                        self.model_path / "model_{}.npz".format(epoch)
                    ).as_posix(),
                    **saveparams
                )
                print("Done.")

        except KeyboardInterrupt:
            pass
        return self

    def transform(
        self,
        x: Union[Tuple[Tuple[str, ...]], Tuple[str, ...]],
        **transform_params
    ) -> np.ndarray:
        # Model
        self._print("Loading model params...")
        if self.last_epoch is not None:
            params = t2v.load_params(
                (
                    self.model_path / "model_{}.npz".format(self.last_epoch)
                ).as_posix()
            )
        else:
            params = t2v.load_params(self._get_last_model().as_posix())

        self._print("Loading dictionaries...")
        with open((self.model_path / "dict.pkl").as_posix(), "rb") as f:
            chardict = pkl.load(f)
        with open((self.model_path / "label_dict.pkl").as_posix(), "rb") as f:
            labeldict = pkl.load(f)
        n_char = len(list(chardict.keys())) + 1
        n_classes = min(len(list(labeldict.keys())) + 1, self.max_classes)

        # iterators
        test_iter = batch.BatchTweets(
            Xt,
            yt,
            labeldict,
            batch_size=settings_char.N_BATCH,
            max_classes=self.max_classes,
            test=True,
        )

        self._print("Building network...")
        # Tweet variables
        tweet = T.itensor3()
        targets = T.imatrix()

        # masks
        t_mask = T.fmatrix()

        # network for prediction
        predictions = self._classify(tweet, t_mask, params, n_classes, n_char)[
            0
        ]
        embeddings = self._classify(tweet, t_mask, params, n_classes, n_char)[
            1
        ]

        # Theano function
        self._print("Compiling theano functions...")
        predict = theano.function([tweet, t_mask], predictions)
        encode = theano.function([tweet, t_mask], embeddings)

        # Test
        self._print("Testing...")
        out_pred = []
        for xr, y in test_iter:
            x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
            p = predict(x, x_m)
            e = encode(x, x_m)
            ranks = np.argsort(p)[:, ::-1]

            for idx, item in enumerate(xr):
                out_pred.append(ranks[idx, :])

        # Save
        return np.asarray(out_pred)

    @classmethod
    def _classify(cls, tweet, t_mask, params, n_classes, n_chars):
        # tweet embedding
        emb_layer = t2v.tweet2vec(tweet, t_mask, params, n_chars)
        # Dense layer for classes
        l_dense = lasagne.layers.DenseLayer(
            emb_layer,
            n_classes,
            W=params["W_cl"],
            b=params["b_cl"],
            nonlinearity=lasagne.nonlinearities.softmax,
        )

        return (
            lasagne.layers.get_output(l_dense),
            l_dense,
            lasagne.layers.get_output(emb_layer),
        )

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _get_last_model(self) -> Path:
        def _get_value(path: Path):
            name = path.with_suffix("").name
            epoch = int(name.split("_")[-1])
            return epoch

        models_paths = list(self.model_path.glob("*.npz"))
        last_model_path = seq(models_paths).sorted(_get_value).first()
        return last_model_path
