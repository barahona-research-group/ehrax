from typing import Literal


CompressionLibLiteral = Literal["blosc", "zlib", "lzo", "bzip2"]
NumericalTypeHint = Literal["B", "N", "O", "C"]  # Binary, Numerical, Ordinal, Categorical
AggregationLiteral = Literal["sum", "or", "w_sum"]
SplitLiteral = Literal["subjects", "admissions", "admissions_intervals"]
TableAggregationLiteral = Literal["admission", "first_admission", "subject"]
