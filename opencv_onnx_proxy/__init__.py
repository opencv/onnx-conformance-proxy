__all__ = []

import sys
import cv2 as cv

from onnx import ModelProto
from onnx.checker import check_model
from onnx.backend.base import Backend, BackendRep
from typing import Any, Tuple


class OpenCVBackendRep(BackendRep):
    def __init__(self, model):
        self.model = model

    def run(self, inputs, **kwargs):  # type: (Any, **Any) -> Tuple[Any, ...]
        if not isinstance(inputs, list):
            inputs = [inputs]

        n = len(inputs)
        input_names = [str(x) for x in range(n)]
        self.model.setInputsNames(input_names)

        for blob, name in zip(inputs, input_names):
            self.model.setInput(cv.Mat(blob), name)

        output_names = self.model.getUnconnectedOutLayersNames()
        ret = self.model.forwardAndRetrieve(output_names)
        assert len(ret) == 1
        return ret[0]


class OpenCVBackend(Backend):
    """
    Implements
    `ONNX's backend API <https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md>`_
    with *OpenCV*.
    """  # noqa: E501

    @classmethod
    def is_compatible(cls, model, device=None, **kwargs):
        """
        Return whether the model is compatible with the backend.

        :param model: unused
        :param device: None to use the default device or a string (ex: `'CPU'`)
        :return: boolean
        """

        return True

    @classmethod
    def is_opset_supported(cls,
                           model,  # type: ModelProto
                           ):
        """
        Return whether the opset for the model is supported by the backend.

        :param model: Model whose opsets needed to be verified.
        :return: boolean and error message if opset is not supported.
        """

        return True, ""

    @classmethod
    def supports_device(cls, device):
        """
        Check whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """

        return True

    @classmethod
    def prepare(cls, model, device=None, **kwargs):
        """
        Load the model and creates a :class:`cv.dnn.Net`
        ready to be used as a backend.

        :param model: ModelProto (returned by `onnx.load`),
            string for a filename or bytes for a serialized model
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :return: :class:`OpenCVBackedRep`
        """

        binary = model.SerializeToString()  # TODO: check if string or bytes
        check_model(binary)
        model_as_list = list(binary)
        net = cv.dnn.readNetFromONNX(buffer=model_as_list)

        return OpenCVBackendRep(net)

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        """
        Compute the prediction.

        :param model: :class:`OpenCVBackedRep` returned
            by function *prepare*
        :param inputs: inputs
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :return: predictions
        """

        # TODO: prepare cant handle this type?
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        '''
        This method is not implemented as it is much more efficient
        to run a whole model than every node independently.
        '''
        raise NotImplementedError("It is much more efficient to run a whole model than every node independently.")


is_compatible = OpenCVBackend.is_compatible
prepare = OpenCVBackend.prepare
run = OpenCVBackend.run_model
supports_device = OpenCVBackend.supports_device
