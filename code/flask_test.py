import requests

student = {
    "Age": 16,
    "Gender": 1,
    "Ethnicity": 2,
    "ParentalEducation": 3,
    "Tutoring": 0,
    "ParentalSupport": 2,
    "Extracurricular": 1,
    "Sports": 0,
    "Music": 1,
    "Volunteering": 0,
    "StudentID": 123,
}

resp = requests.post("http://localhost:9696/predict", json=student)
print(f'Result for Student ID: {student["StudentID"]}')
print(resp.json())
