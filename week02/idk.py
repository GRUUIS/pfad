import os
import time
import random

def clear():
	# Works for Windows and Unix
	os.system('cls' if os.name == 'nt' else 'clear')

def shake_text(text, shakes=15, delay=0.2):
	for _ in range(shakes):
		x = random.randint(0, 20)
		y = random.randint(0, 15)
		clear()
		print("\n" * y + " " * x + text)
		time.sleep(delay)

if __name__ == "__main__":
	shake_text("idk")
# print(type("20"))
# print(type(int("20")))