from .trace_core import TRACECore


class TRACETrainer(TRACECore):
    """Lightning trainer for AIG predictive TRACE."""

    def __init__(self, args):
        args.encoder_type = 'TRACE'
        args.aggr = 'transformer'
        super().__init__(args)
