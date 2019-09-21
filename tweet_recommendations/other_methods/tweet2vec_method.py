"""Provides implementation of adapter for tweet2vec method.

The method descriptions is available at: https://arxiv.org/abs/1605.03481
The original code is available at: https://github.com/bdhingra/tweet2vec
"""
import os

os.environ["THEANO_FLAGS"] = "floatX=float32,device=cuda0"

import pickle as pkl
import string
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import lasagne
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from functional import seq

from tweet_recommendations.other_methods.method import Method
from .tweet2vec.tweet2vec import batch_char as batch, settings_char, t2v

POLISH_CHARACTERS = "ąćęłńóśżź"
VALID_CHARACTERS = (
    string.printable + POLISH_CHARACTERS + POLISH_CHARACTERS.upper()
)


class Tweet2Vec(Method):
    """Adapter for tweet2vec method in scikit-learn manner."""
    def __init__(
        self,
        epochs: int,
        model_path: str,
        verbose: bool = False,
        last_epoch: Optional[int] = None,
    ):
        """Instantiates the class.

        :param epochs: Number of epochs that the model will be trained for once
            the `fit` method is called.
        :param model_path: Path to the folder where all auxiliary files will
            be saved, such as class mappings, weights etc. If the directory
            doesn't exists, then it creates one. Also, overwrites all the
            content in the existing directory during `fit` method. It is used
            also to load weights from if the directory contains an appropriate
            file.
        :param verbose: Whether method should be verbose and print intermediate
            results.
        :param last_epoch: Last epoch of training that the learning should
            be continued from. Also, it used to point a particular file of many
            that are saved every epoch during epoch. If the value is None or
            is less then 0, then weights from the last epoch will be loaded.
        """
        self.model_path = Path(model_path)
        self.epochs = epochs
        self.verbose = verbose
        self.max_classes = 999999
        self.last_epoch = last_epoch
        self.params = None

        self.n_char = None
        self.n_classes = None
        self.chardict = None
        self.labeldict = None
        self.classnum_to_label_map = None
        self.predict = None
        self.encode = None
        self.net = None
        self.train = None
        self.cost_val = None

        self.tweet_input = None
        self.targets_input = None
        self.t_mask_input = None

    def _create_classnum_to_label_map(self):
        """Create mapping from an index to a label (hashtag)."""
        self.classnum_to_label_map = np.ndarray(
            (len(self.labeldict) + 1,), dtype=np.object
        )
        for key, index in self.labeldict.items():
            self.classnum_to_label_map[index] = key

    def _load_model(self):
        """Load all necessary data so the model can work properly.

        This includes weights of the model, a dict with mapping from a single
        character to its class and a dict with mappings from hashtags to
        appropriate intiger classes. If the `last_epoch` is provided, then
        weights from file `model_{last_epochs}.npz` are loaded from the given
        model directory.
        """
        self._print("Loading model params...")
        with open((self.model_path / "dict.pkl").as_posix(), "rb") as f:
            self.chardict = pkl.load(f)
        with open((self.model_path / "label_dict.pkl").as_posix(), "rb") as f:
            self.labeldict = pkl.load(f)
        self._create_classnum_to_label_map()
        self.n_char = len(list(self.chardict.keys())) + 1
        self.n_classes = min(
            len(list(self.labeldict.keys())) + 1, self.max_classes
        )
        if self.last_epoch is not None:
            self.params = t2v.load_params(
                (
                    self.model_path / "model_{}.npz".format(self.last_epoch)
                ).as_posix()
            )
        else:
            self.params = t2v.load_params(self._get_last_model().as_posix())

    def _build_network(self):
        """Build network, including inputs, weights and the whole structure."""
        # Tweet variables
        self.tweet_input = T.itensor3()
        self.targets_input = T.ivector()
        self.t_mask_input = T.fmatrix()

        self.params = t2v.init_params(n_chars=self.n_char)
        # classification params
        self.params["W_cl"] = theano.shared(
            np.random.normal(
                loc=0.,
                scale=settings_char.SCALE,
                size=(settings_char.WDIM, self.n_classes),
            ).astype("float32"),
            name="W_cl",
        )
        self.params["b_cl"] = theano.shared(
            np.zeros((self.n_classes,)).astype("float32"), name="b_cl"
        )
        # network for prediction
        predictions, net, embeddings = self._classify(
            self.tweet_input,
            self.t_mask_input,
            self.params,
            self.n_classes,
            self.n_char,
        )

        # Theano function
        self._print("Compiling theano functions...")
        self.predict = theano.function(
            [self.tweet_input, self.t_mask_input], predictions
        )
        self.encode = theano.function(
            [self.tweet_input, self.t_mask_input], embeddings
        )
        self.net = net
        self._print("Building network...")

        # batch loss
        loss = lasagne.objectives.categorical_crossentropy(
            predictions, self.targets_input
        )
        cost = T.mean(
            loss
        ) + settings_char.REGULARIZATION * lasagne.regularization.regularize_network_params(
            self.net, lasagne.regularization.l2
        )
        cost_only = T.mean(loss)

        # params and updates
        self._print("Computing updates...")
        lr = settings_char.LEARNING_RATE
        mu = settings_char.MOMENTUM
        updates = lasagne.updates.nesterov_momentum(
            cost, lasagne.layers.get_all_params(self.net), lr, momentum=mu
        )

        # Theano function
        self._print("Compiling theano functions...")

        inps = [self.tweet_input, self.t_mask_input, self.targets_input]
        self.cost_val = theano.function(inps, [cost_only, embeddings])
        self.train = theano.function(inps, cost, updates=updates)

    def fit(
        self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params
    ) -> "Method":
        """Fit parameters of the method.

        It creates a directory where files will be saved, preprocesses
        dataset appropriately, builds the network and fits parameters of that
        network. All the intermediate results are saved in the given in the
        constructor path. The method can be stopped with Ctrl+C combination -
        then the program execution will proceed further.
        :param x: Frame containing lemmas under 'lemmas' key, where a single
            entry is a list of lemmas. It must contain a key 'hashtags' with
            ground truth hashtags, as a list list of dicts.
        :param y: None, needed for a compatibility.
        :param fit_params: Additional parameters for the network.
        :return: self.
        """
        self._print("Building Model...")
        if not self.model_path.exists():
            self.model_path.mkdir(parents=True, exist_ok=True)

        if y is None:
            y = self._extract_hashtags(x)
        self.max_classes = len(np.unique(y))

        # Training data
        Xt = x["lemmas"].apply(lambda a_tweet: " ".join(a_tweet)).tolist()
        Xt = self._preprocess_lemmas(Xt)
        Xt, y = self._split_multiple_tags_to_separate_entries(Xt, y)

        # Build dictionaries from training data
        self.chardict, charcount = batch.build_dictionary(Xt)
        self.n_char = len(list(self.chardict.keys())) + 1
        batch.save_dictionary(
            self.chardict, charcount, (self.model_path / "dict.pkl").as_posix()
        )

        self.labeldict, labelcount = batch.build_label_dictionary(y)
        batch.save_dictionary(
            self.labeldict,
            labelcount,
            (self.model_path / "label_dict.pkl").as_posix(),
        )
        self._create_classnum_to_label_map()

        self.n_classes = min(
            len(list(self.labeldict.keys())) + 1, self.max_classes
        )

        self._build_network()
        if self.last_epoch >= 0:
            self._load_model()

        # iterators
        train_iter = batch.BatchTweets(
            Xt,
            y,
            self.labeldict,
            batch_size=settings_char.N_BATCH,
            max_classes=self.max_classes,
        )

        self._print("Building network...")
        # batch loss
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
                    x, x_m = batch.prepare_data(
                        xr, self.chardict, n_chars=self.n_char
                    )
                    if x is None:
                        self._print(
                            "Minibatch with zero samples under maxlength."
                        )
                        uidx -= 1
                        continue

                    curr_cost = self.train(x, x_m, y)
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
                    for kk, vv in self.params.items():
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
                for kk, vv in self.params.items():
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
        x: Union[Tuple[Tuple[str, ...], ...], Tuple[str, ...]],
        **transform_params
    ) -> np.ndarray:
        """Predict most appropriate hashtags for each entry hashtag.

        If the model was not fit before this call, then the model is loaded
        from a given in the constructor path.
        :param x: List of tweets contents or list of lists, where each sublist
            contains single words. The words should be lemmas but the method
            works fine without it.
        :param transform_params: Additional transformation parameters for the
            method.
        :return: 2D numpy array of predicted hashtags, sorted from the most
            probable to the least probable for each entry tweet.
        """
        # Model
        if self.params is None:
            self._build_network()
            self._load_model()

        if type(x[0]) in [tuple, list, np.ndarray]:
            x = [" ".join(entry) for entry in x]

        # iterators
        test_iter = batch.BatchTweets(
            x,
            None,
            self.labeldict,
            batch_size=settings_char.N_BATCH,
            max_classes=self.max_classes,
            test=True,
        )

        # Test
        self._print("Testing...")
        out_pred = []
        for xr, y in test_iter:
            x, x_m = batch.prepare_data(xr, self.chardict, n_chars=self.n_char)
            p = self.predict(x, x_m)
            ranks = np.argsort(p)[:, ::-1]

            for idx, item in enumerate(xr):
                out_pred.append(ranks[idx, :])

        output = np.asarray(out_pred)
        labels = self.classnum_to_label_map[output]

        return labels

    @classmethod
    def _classify(cls, tweet, t_mask, params, n_classes, n_chars):
        """Create further model parameters - the head of the network."""
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
        """Find the last trained model from a given directory.

        Assumes that there are multiple models that were saved during training
        process.
        """
        def _get_value(path: Path):
            name = path.with_suffix("").name
            epoch = int(name.split("_")[-1])
            return epoch

        models_paths = list(self.model_path.glob("*.npz"))
        last_model_path = seq(models_paths).sorted(_get_value).first()
        return last_model_path

    @classmethod
    def _extract_hashtags(
        cls, original_frame: pd.DataFrame
    ) -> List[List[str]]:
        """Extract hashtags from nested structure of ground truth data."""
        return [
            [tag["text"] for tag in entry]
            for entry in original_frame["hashtags"]
        ]

    @classmethod
    def _preprocess_lemmas(cls, tweets_lemmas: List[str]) -> List[str]:
        """Filter tweets' contents so it contains reasonable chars.

        The method casts each char in the tweet to a one hot vector. Since
        the original lemmas can contain emojis, which are abundant, the method
        would produce huge vectors without this filtering.
        """
        output = []
        for tweet in tweets_lemmas:
            output.append(
                seq(list(tweet))
                    .filter(lambda char: char in VALID_CHARACTERS)
                    .to_list()
            )
        return output

    @classmethod
    def _split_multiple_tags_to_separate_entries(
        cls, lemmas: List[str], tags: List[List[str]]
    ) -> Tuple[List[str], List[str]]:
        """Repeat tweets with multiple tags and assign each tag to separate.

        The method assumes that for each entry there is a one tag only.
        Otherwise, it would be hard to define output of the network, since
        we want to use softmax activation function in the end.
        """
        new_x = []
        new_y = []
        for x_sample, y_sample in zip(lemmas, tags):
            for tag in y_sample:
                new_x.append(x_sample)
                new_y.append(tag)
        return new_x, new_y
