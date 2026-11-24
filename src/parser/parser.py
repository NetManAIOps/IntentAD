import re
import ast
from typing import List, Tuple, Any, Optional


class LLMOutputParser:
	"""
	Parser for LLM anomaly detection output. Extracts anomaly positions and pattern types.
	Expected output format:
	[Anomaly Positions]
	[(start, end, pattern_type), ...]
	"""
	# POSITIONS_REGEX = r"\[Anomaly Positions\]\s*\n\s*(\[.*?\])"
	# POSITIONS_REGEX = r"(?:\[Anomaly Positions\]|##\s*Anomaly Positions)\s*\n\s*(\[.*?\])"
	# new regex, output format: <answer>\n[(257, 261, 'dip', 746.0)]\n</answer>
	POSITIONS_REGEX = r"(\[.*?\])"

	@classmethod
	def _extract_raw_list(cls, text: str) -> Optional[str]:
		# matches = re.findall(cls.POSITIONS_REGEX, text, re.DOTALL)
		# if len(matches) > 1:
		# 	return matches[-1]
		# return None
		match = re.search(cls.POSITIONS_REGEX, text, re.DOTALL)
		if not match:
			return None
		return match.group(1)

	@classmethod
	def parse(cls, output_text: str, format: str = "interval", answer_tag: bool = True) -> Tuple[List[Tuple[int, int]], List[Any]]:
		"""
		Parse the LLM output and return:
		- List of (start, end) tuples for anomaly positions
		- List of pattern types for each anomaly
		"""
		if answer_tag:
			# Extract content within <answer> tags
			answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL)
			if not answer_match:
				return [], [], []
			answer = answer_match.group(1).strip()
		else:
			answer = output_text.strip()
		raw = cls._extract_raw_list(answer)
		if not raw:
			return [], [], []
		try:
			data = ast.literal_eval(raw)
		except (ValueError, SyntaxError):
			return [], [], []
		positions: List[Tuple[int, int]] = []
		patterns: List[Any] = []
		magnitudes: List[float] = []
		# Iterate over raw items; ensure output lists align with input length
		if format == "interval":
			for item in data:
				# Default values if parsing fails
				default_pos = (-1, -1)
				default_pattern = None
				default_magnitude = float('inf')
				if not isinstance(item, (list, tuple)):
					positions.append(default_pos)
					patterns.append(default_pattern)
					magnitudes.append(default_magnitude)
					continue
				elif len(item) < 2:
					positions.append(default_pos)
					patterns.append(default_pattern)
					magnitudes.append(default_magnitude)
					continue
				elif len(item) < 3:
					start, end, pattern, magnitude = item[0], item[1], default_pattern, default_magnitude
				elif len(item) < 4:
					# start, end, pattern, magnitude = item[0], item[1], item[2], None
					start, end, pattern, magnitude = item[0], item[1], item[2], default_magnitude
				else:
					start, end, pattern, magnitude = item[0], item[1], item[2], item[3]
				try:
					start_i = int(start)
					end_i = int(end)
					pattern = str(pattern) if pattern is not default_magnitude else default_magnitude
					magnitude_f = float(magnitude) if magnitude is not default_magnitude else default_magnitude
				except (ValueError, TypeError):
					print(f"[LLMOutputParser] Warning: Failed to parse item {item}, using default values, raw answer: {answer}")
					positions.append(default_pos)
					patterns.append(default_pattern)
					magnitudes.append(default_magnitude)
					continue
				if magnitude_f is default_magnitude:
					print(f"[LLMOutputParser] Warning: None magnitude, raw answer: {answer}")
				positions.append((start_i, end_i))
				patterns.append(pattern)
				magnitudes.append(magnitude_f)
		elif format == "index":
			for item in data:
				# Default values if parsing fails
				default_pos = (-1, -1)
				default_pattern = None
				default_magnitude = None
				positions.append((item, item))
				patterns.append(default_pattern)
				magnitudes.append(default_magnitude)
		else:
			raise Exception("Unknown output format")
		return positions, patterns, magnitudes
