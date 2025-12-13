print("Testing detector import...")
from detector import ThreatDetector

print("Creating detector...")
detector = ThreatDetector()

print("Loading models...")
detector.load()

print("SUCCESS - detector loaded")
