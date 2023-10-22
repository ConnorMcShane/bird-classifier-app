import requests


url_dict = {
    "0": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg",
    "1": "https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg",
    "2": "https://upload.wikimedia.org/wikipedia/commons/8/81/Eumomota_superciliosa.jpg",
    "3": "https://i.pinimg.com/originals/f3/fb/92/f3fb92afce5ddff09a7370d90d021225.jpg",
    "4": "https://cdn.britannica.com/77/189277-004-0A3BC3D4.jpg"
}


server = "http://127.0.0.1:8000"


if __name__ == "__main__":
    
    for _ in range(10):
        return_dict = requests.post(f"{server}/classify", json={"data": url_dict}).json()
        print("Classificaton request returned")