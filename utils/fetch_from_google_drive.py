import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]


def fetch_first_ten_files():
  creds = None
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    service = build("drive", "v3", credentials=creds)

    # Call the Drive v3 API
    results = (
        service.files()
        .list(pageSize=10, fields="nextPageToken, files(id, name)")
        .execute()
    )
    items = results.get("files", [])

    if not items:
      print("No files found.")
      return
    print("Files:")
    for item in items:
      print(f"{item['name']} ({item['id']})")
  except HttpError as error:
    # TODO(developer) - Handle errors from drive API.
    print(f"An error occurred: {error}")


import requests
import io
from libs.foxutils.utils.core_utils import mkdir_if_not_exist


def fetch_h5_file_from_drive(url, savename="dataset.hd5"):
    with requests.Session() as session:
        r = session.get(url, stream=True)
        r.raise_for_status()
        mkdir_if_not_exist(savename)
        with open(savename, "wb") as hd5:
            for chunk in r.iter_content(chunk_size=io.DEFAULT_BUFFER_SIZE):
                hd5.write(chunk)


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def authenticate_google_drive():
    gauth = GoogleAuth(settings_file="config_files/pydrive_settings.yaml")
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    return drive


def download_file_from_drive(drive, file_id, local_path):
    file_obj = drive.CreateFile({"id": file_id})
    mkdir_if_not_exist(local_path)
    file_obj.GetContentFile(local_path)
    #print(f"Downloaded: {file_obj['title']} at {local_path}")


def load_weights_from_google_drive(weights_file_id, local_path):
    drive = authenticate_google_drive()
    download_file_from_drive(drive, weights_file_id, local_path)