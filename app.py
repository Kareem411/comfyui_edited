from typing import Union, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from ComfyUI.Python_script.edited_workflow import main
import os
import json
import asyncio
from azure.storage.blob import BlobServiceClient
import uuid
from io import BytesIO
import numpy as np
from PIL import Image


html = """
<!DOCTYPE html>
<html>
<head>
    <title>Websocket Demo</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
</head>
<body>
<div class="container mt-3">
    <h1>FastAPI WebSocket Chat</h1>
    <h2>Your ID: <span id="ws-id"></span></h2>
    <form action="" onsubmit="sendMessage(event)">
        <label for="ipadapterImage">IPAdapter Image:</label>
        <input type="text" class="form-control" id="ipadapterImage" autocomplete="off"/>
        
        <label for="inputImage">Input Image:</label>
        <input type="text" class="form-control" id="inputImage" autocomplete="off"/>
        
        <label for="facePrompt">Face Prompt:</label>
        <input type="text" class="form-control" id="facePrompt" autocomplete="off"/>
        
        <label for="initialPrompt">Initial Prompt:</label>
        <input type="text" class="form-control" id="initialPrompt" autocomplete="off"/>
        
        <button class="btn btn-outline-primary mt-2">Send</button>
    </form>
    <ul id='messages' class="mt-5"></ul>
</div>

<script>
    var client_id = Date.now()
    document.querySelector("#ws-id").textContent = client_id;
    var ws = new WebSocket(`ws://localhost:8000/ws/${client_id}`);
    ws.onmessage = function(event) {
        var messages = document.getElementById('messages')
        var message = document.createElement('li')
        var content = document.createTextNode(event.data)
        message.appendChild(content)
        messages.appendChild(message)
    };

    function sendMessage(event) {
        var ipadapterImage = document.getElementById("ipadapterImage").value;
        var inputImage = document.getElementById("inputImage").value;
        var facePrompt = document.getElementById("facePrompt").value;
        var initialPrompt = document.getElementById("initialPrompt").value;

        var data = {
            "ipadapter_input": ipadapterImage,
            "image_input": inputImage,
            "prompt_1": initialPrompt,
            "prompt_2": facePrompt
        };

        ws.send(JSON.stringify(data));
        event.preventDefault();
    }
</script>
</body>
</html>
"""


app = FastAPI()


@app.post("/process-data")
async def process_data(data: Union[dict, None] = None):
    if data is None:
        raise HTTPException(status_code=400, detail="Invalid data provided")

    # Convert the received data to a dictionary
    data_dict = jsonable_encoder(data)

    # Call the main function from workflow_api.py with the provided data
    try:
        generated_images, filenames = main(data_dict)
        return JSONResponse(content={"message": "Processing completed", "Generated Images": generated_images, "Image names":filenames}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": f"Error processing data: {str(e)}"}, status_code=500)





class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def send_progress_messages(self, websocket: WebSocket, progress_generator):
        try:
            async for message in progress_generator:
                await websocket.send_text(message)
        except WebSocketDisconnect:
            pass

manager = ConnectionManager()




@app.get("/")
async def get():
    return HTMLResponse(html)

container_name = "photos"
connection_str = "DefaultEndpointsProtocol=https;AccountName=comfyuiimages;AccountKey=WBeQ6FeRkaHVZyZJoM0h6BSvb615urc8bbLt8Bw007WJI6Z9/Yj/QuVGhe+P7DxaHgFrEUrZUAmc+ASt4VgYzQ==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_str) # create a blob service client to interact with the storage account
try:
    container_client = blob_service_client.get_container_client(container=container_name) # get container client to interact with the container in which images will be stored
    container_client.get_container_properties() # get properties of the container to force exception to be thrown if container does not exist
except Exception as e:
    container_client = blob_service_client.create_container(container_name) # create a container in the storage account if it does not exist

# New WebSocket endpoint for handling data from the form
@app.websocket("/ws/{client_id}")
async def data_websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        async for data in websocket.iter_text():
            try:
                data_dict = json.loads(data)

                # Start the main function and get the progress generator
                progress_queue = asyncio.Queue()
                progress_generator = main(data_dict, progress_queue)

                # Send progress messages in real-time
                async for progress_message in progress_generator:
                    await websocket.send_text(str(progress_message))

                # Once the main function is over, get the final result
                generated_images = await progress_queue.get()

                # Send the final result to the client
                response_data = {
                    "status": "success",
                    "message": "Main function completed.",
                    "generated_images": generated_images,
                }
                filenames = [image["filename"] for ui_data in generated_images for image in ui_data.get('ui', {}).get('images', [])]
                images_directory = os.path.join(os.getcwd(), "ComfyUI", "output")
                file_paths = [os.path.join(images_directory, filename) for filename in filenames]
                images_url = []
                for file_path in file_paths:
                    with open(file_path, "rb") as data:
                        # Upload the blob
                        blob_prefix = "image_"
                        blob_name = f"{blob_prefix}{str(uuid.uuid4())}.png"
                        container_client.upload_blob(name=blob_name, data=data)
                        images_url.append(container_client.get_blob_client(blob_name).url)
                print(images_url)
                await websocket.send_text(json.dumps(images_url))

            except WebSocketDisconnect:
                # Handle the WebSocketDisconnect here
                break

    except WebSocketDisconnect:
        pass  # Already handled disconnect, just pass

    finally:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} has left the chat")








# create websocket connection
# when the websocket is live, listen to client events
# when client sends event that triggers comfyui, call the main() in workflow_api here
# send back the logs of workflow_api (image generation process) back to the client constantly, and you can do that bc the connection is live
# close the websocket once the image generation process is done.