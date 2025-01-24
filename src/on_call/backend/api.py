from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

import time

app = FastAPI()

origins = [
    "http://localhost:5173"  # frontend port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def event_stream():
    # Sends an event every second with data: "Message {i}"
    for i in range(10):
        event_str = "event: stream_event"
        data_str = f"data: Message {i}"
        yield f"{event_str}\n{data_str}\n\n"
        time.sleep(1)


@app.get("/stream")
async def stream():
    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)
