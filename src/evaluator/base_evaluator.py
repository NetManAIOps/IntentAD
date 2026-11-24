from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEvaluator(ABC):
	"""
	Abstract base class for all evaluators.
	"""
	@abstractmethod
	def evaluate(self, y_true: Any, y_pred: Any, **kwargs) -> Dict[str, Any]:
		"""
		Evaluate predictions against ground truth.
		"""
		pass
