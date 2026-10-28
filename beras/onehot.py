import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        
        labels = np.unique(data)
        one_hot_vectors = np.eye(len(labels))

        self.vector_label_mapping = {}
        
        for index, label in enumerate(labels):
            self.vector_label_mapping[label] = one_hot_vectors[index]

    def forward(self, data):
        if not hasattr(self, "vector_label_mapping"):
            self.fit(data)
        
        encoded_result = []

        for label in data:
            encoded_result.append(self.vector_label_mapping[label])
        
        return np.asarray(encoded_result)

    def inverse(self, data):
        labels = list(self.vector_label_mapping.keys())
        #values are either 0 and only one is 1, so argmax will give indices with 1 
        hot_indices = np.argmax(data, axis = 1)

        decoded_result = []

        for index in hot_indices:
            decoded_result.append(labels[index])

        return np.asarray(decoded_result) 
