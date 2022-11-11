import whisper

model = whisper.load_model("base")
out = model.transcribe("mp3 goes here", language="en")
out