import tkinter as tk
import threading
import whisper
import numpy as np
import pyaudio
from difflib import SequenceMatcher

# Initialize the Whisper model with the English-only version
model = whisper.load_model("base.en")

# Variables for controlling the teleprompter
listening = False
current_line_index = 0
teleprompter_text = ("Your teleprompter text goes here.\n"
                     "This is a sample text to demonstrate scrolling and highlighting.\n"
                     "Please replace this with your actual content.")

# Function to update the teleprompter display and highlight the current line
def update_displayed_text():
    global teleprompter_text, current_line_index
    teleprompter_label.config(state=tk.NORMAL)
    teleprompter_label.delete(1.0, tk.END)
    teleprompter_label.insert(tk.END, teleprompter_text)

    lines = teleprompter_text.splitlines()
    if current_line_index < len(lines):
        start_index = f"{current_line_index + 1}.0"
        end_index = f"{current_line_index + 1}.end"
        teleprompter_label.tag_remove("highlight", 1.0, tk.END)
        teleprompter_label.tag_add("highlight", start_index, end_index)
        teleprompter_label.tag_configure("highlight", background="yellow")
        teleprompter_label.see(start_index)  # Scroll to the highlighted line
    teleprompter_label.config(state=tk.DISABLED)

# Function to update the teleprompter text from the input
def update_text():
    global teleprompter_text, current_line_index
    teleprompter_text = text_input.get("1.0", "end-1c")
    current_line_index = 0
    update_displayed_text()

# Function to check if two strings are similar enough
def is_similar(a, b):
    return SequenceMatcher(None, a, b).ratio() > 0.6  # 60% similarity threshold

# Function to continuously listen for voice commands
def listen_for_commands():
    global current_line_index  # Ensure access to the global variable
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)  # Slightly larger buffer size

    while listening:
        frames = []
        try:
            # Process audio in slightly larger chunks for better accuracy
            for _ in range(int(16000 / 1024 * 4)):  # 4 seconds of audio
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(np.frombuffer(data, np.int16))

            audio_data = np.concatenate(frames).astype(np.float32)
            audio_data /= np.max(np.abs(audio_data))

            # Transcribe audio using Whisper for English only
            result = model.transcribe(audio_data, fp16=False, language="en")
            recognized_text = result['text'].strip().lower()
            print(f"Recognized text: {recognized_text}")

            # Get the current line in the teleprompter text
            lines = teleprompter_text.splitlines()
            if current_line_index < len(lines):
                current_line = lines[current_line_index].strip().lower()
                print(f"Current line: {current_line}")

                # Check if the recognized text is similar to the current line
                if is_similar(current_line, recognized_text):
                    current_line_index += 1
                    update_displayed_text()
                else:
                    print("No match found.")

        except Exception as e:
            print(f"Error reading audio: {e}")

    stream.stop_stream()
    stream.close()
    p.terminate()

# Function to toggle listening
def toggle_listening():
    global listening
    listening = not listening
    if listening:
        listen_button.config(text="Stop Listening")
        # Start the listening thread if not already started
        voice_thread = threading.Thread(target=listen_for_commands, daemon=True)
        voice_thread.start()
    else:
        listen_button.config(text="Start Listening")

# Set up the Tkinter window
root = tk.Tk()
root.title("Teleprompter")

# Add a Scrollbar and Text widget for the teleprompter display
teleprompter_frame = tk.Frame(root)
teleprompter_frame.pack(fill=tk.BOTH, expand=True, pady=20)

scrollbar = tk.Scrollbar(teleprompter_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

teleprompter_label = tk.Text(teleprompter_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, height=10, state=tk.DISABLED)
teleprompter_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=teleprompter_label.yview)

text_input = tk.Text(root, height=5, width=50)
text_input.pack()

update_button = tk.Button(root, text="Update Text", command=update_text)
update_button.pack(pady=10)

listen_button = tk.Button(root, text="Start Listening", command=toggle_listening)
listen_button.pack(pady=10)

# Initialize the display with the default text
update_displayed_text()

# Run the Tkinter main loop
root.mainloop()
