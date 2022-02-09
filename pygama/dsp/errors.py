class DSPError(Exception):
    """Base class for signal processors"""
    pass

class DSPFatal(DSPError):
    """Fatal error thrown by DSP processors that halts production
        Attributes:
        wf_range: range of wf indices. This will be set after the exception
                  is caught, and appended to the error message
        processor: string of processor and arguments. This will be set after
                   the exception is caught, and appended to the error message
    """
    def __init__(self, *args):
        super(DSPFatal, self).__init__(*args)
        self.wf_range = None
        self.processor = None

    def __str__(self):
        suffix = ''
        if self.wf_range:
            suffix += '\nThrown while processing entries ' + str(self.wf_range)
        if self.processor:
            suffix += '\nThrown by ' + self.processor
        return super(DSPFatal, self).__str__() + suffix

class ProcessingChainError(DSPError):
    """Error thrown when there is a problem setting up a processing chain"""
    pass
