from datetime import datetime, timedelta
import requests
import os

BASE_DIR = "./images"
BASE_URL = "https://maps.consumer-digital.api.metoffice.gov.uk/wms_ob/single/high-res/rainfall_radar/"

os.makedirs(BASE_DIR, exist_ok=True)

# ----------------------------------
# Find next folder number
# ----------------------------------
existing = [
    int(name)
    for name in os.listdir(BASE_DIR)
    if name.isdigit() and os.path.isdir(os.path.join(BASE_DIR, name))
]

next_folder_num = max(existing) + 1 if existing else 0
folder_name = os.path.join(BASE_DIR, str(next_folder_num))
os.makedirs(folder_name, exist_ok=True)

print(f"Saving to folder: {folder_name}")

# ----------------------------------
# Time range: last 48 hours
# ----------------------------------
now = datetime.utcnow()
now = now - timedelta(
    minutes=now.minute % 15, seconds=now.second, microseconds=now.microsecond
)

start = now - timedelta(hours=48)

print(f"Downloading radar images from {start} to {now}")

# ----------------------------------
# Download loop
# ----------------------------------
count = 0
missing = 0

while start <= now:
    url = BASE_URL + start.strftime("%Y-%m-%dT%H:%M:%SZ") + ".png"
    print(url)

    try:
        response = requests.get(url, timeout=20)
    except Exception as e:
        print("Request error:", e)
        missing += 1
        start += timedelta(minutes=15)
        continue

    if response.status_code == 200:
        filename = os.path.join(folder_name, f"{count}.png")
        with open(filename, "wb") as f:
            f.write(response.content)
        count += 1
    else:
        print(f"Missing frame ({response.status_code})")
        missing += 1

    start += timedelta(minutes=15)

print("\nDone.")
print("Downloaded:", count)
print("Missing:", missing)
print("Saved in:", folder_name)
