from typing import Any, Dict, List
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from .base_evaluator import BaseEvaluator


class PatternEvaluator(BaseEvaluator):
	"""
	Evaluator for anomaly pattern classification.
	"""
	def __init__(self, average: str = 'binary', labels: List[Any] = None):
		self.average = average
		self.labels = labels

	def evaluate(self, y_true: List[Any], y_pred: List[Any], **kwargs) -> Dict[str, Any]:
		"""
		Evaluate classification predictions.
		"""
		precision, recall, f1, support = precision_recall_fscore_support(
			y_true, y_pred, average=self.average, labels=self.labels, zero_division=0
		)
		cm = confusion_matrix(y_true, y_pred, labels=self.labels)
		return {
			'precision': float(precision),
			'recall': float(recall),
			'f1': float(f1),
			'support': support.tolist() if hasattr(support, 'tolist') else support,
			'confusion_matrix': cm.tolist()
		}
