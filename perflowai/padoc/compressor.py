from .trace import BaseTrace


class Compressor:
    
    def compress(self, trace: BaseTrace) -> BaseTrace:
        raise NotImplementedError

    def decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        raise NotImplementedError


class TemplateCompressor(Compressor):

    def compress(self, trace: BaseTrace) -> BaseTrace:
        # TODO
        pass

    def decompress(self, compressed_trace: BaseTrace) -> BaseTrace:
        # TODO
        pass
