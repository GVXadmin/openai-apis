import datetime
from typing import Dict, Any

user_appointment: Dict[str,Any] = {}

def validate_date(user_date_str: str, current_date_str: str) -> Dict[str, Any]:
    try:
        user_date = datetime.datetime.strptime(user_date_str, "%m/%d/%Y").date() #format: MM/DD/YYYY
    except ValueError:
        return {
            "Message": "Invalid date format. Please enter the date in MM/DD/YYYY format.",
            "question_id": "preferred_date",
            "input_type": "text",
            "input_format": "mm/dd/yyyy"
        }
    current_date = datetime.datetime.strptime(current_date_str, "%m/%d/%Y").date()
    diff = (user_date - current_date).days
    if user_date < current_date:
        return {"Message": "The selected date has already passed. Please choose a future date.", "question_id": "preferred_date"}
    if diff < 7:
        return {"Message": "Appointments must be scheduled at least 7 days in advance. Please select a later date.", "question_id": "preferred_date"}
    if diff > 21:
        return {"Message": "Appointments can only be scheduled up to 21 days in advance. Please select an earlier date.", "question_id": "preferred_date"}
    return None

def get_service_type():
    return {
        "Message": "Sure, I can help you schedule your appointment. Please choose the type of coaching session you'd like.",
        "question_id": "service_type",
        "input_type": "options",
        "Options": [
            {"Id": 1, "Option": "Group Coaching"},
            {"Id": 2, "Option": "Individual Coaching"},
            {"Id": 3, "Option": "Nutrition Coaching"}
        ]
    }

def get_preferred_time():
    return {
        "Message": "When would you like to schedule your session? Select all that apply.",
        "question_id": "preferred_time",
        "Prompt": "",
        "input_type": "checkbox",
        "Options": [
            {
            "Id": 1,
            "Option": "Monday AM"
             },
        {
            "Id": 2,
            "Option": "Monday PM"
        },
        {
            "Id": 3,
            "Option": "Tuesday AM"
        },
        {
            "Id": 4,
            "Option": "Tuesday PM"
        },
        {
            "Id": 5,
            "Option": "Wednesday AM"
        },
        {
            "Id": 6,
            "Option": "Wednesday PM"
        },
        {
            "Id": 7,
            "Option": "Thursday AM"
        },
        {
            "Id": 8,
            "Option": "Thursday PM"
        },
        {
            "Id": 9,
             "Option": "Friday AM"
        },
        {
            "Id": 10,
            "Option": "Friday PM"
    }
  ]
}

def get_preferred_date():
    return {
        "Message": "Do you have a specific date in mind? Please enter the date in MM/DD/YYYY format.",
        "question_id": "preferred_date",
        "Prompt": "",
        "input_type": "text",
        "input_format": "mm/dd/yyyy"
    }

def get_special_requirement():
    return {
        "Message": "Do you have any special requests for your appointment?",
        "question_id": "special_requirement",
        "Prompt": "",
        "Options": [
            {
                "Id": 1,
                "Option": "None"
            },
            {
                "Id": 2,
                "Option": "Interpreter Services"
            },
            {
                "Id": 3,
                "Option": "Other"
            }
        ]
    }

def get_special_requirement_other():
    return {
        "message": "Please describe your special request.",
        "question_id": "special_requirement_other",
        "input_type": "text",
        "options": []
    }

def handle_appointment_workflow(user_response: str): 
    global user_appointment

    if user_response.lower() in ["book an appointment", "schedule an appointment"]:
        user_appointment.clear()  
        return get_service_type()
    
    elif "service_type" not in user_appointment:
        valid_services = ["group coaching", "individual coaching", "nutrition coaching"]
        if user_response.lower() not in valid_services:
            return get_service_type()
        user_appointment["service_type"] = user_response
        return get_preferred_time()
    
    elif "preferred_time" not in user_appointment:
        valid_times = [
        "monday am", "monday pm", "tuesday am", "tuesday pm",
        "wednesday am", "wednesday pm", "thursday am", "thursday pm",
        "friday am", "friday pm"
    ]
        if user_response.lower() not in valid_times:
            return get_preferred_time()  
        user_appointment["preferred_time"] = user_response
        return get_preferred_date()
    
    elif "preferred_date" not in user_appointment:
        current_date = datetime.datetime.today().strftime("%m/%d/%Y")
        validated_date = validate_date(user_response, current_date)
        if validated_date:
            return validated_date  
        user_appointment["preferred_date"] = user_response
        return get_special_requirement()
    
    elif "special_requirement" not in user_appointment:
        valid_requirements = ["none", "interpreter services", "other"]
        if user_response.lower() not in valid_requirements:
            return get_special_requirement()  
        user_appointment["special_requirement"] = user_response
        if user_response.lower() == "other":
            return get_special_requirement_other() 
    
    elif "special_requirement_other" not in user_appointment and user_appointment["special_requirement"] == "Other":
        user_appointment["special_requirement_other"] = user_response
        return finalize_appointment() 

    return finalize_appointment()



def finalize_appointment():
    global user_appointment
    
    # Store the final response
    final_response = {
        "Message": "Thank you! Your appointment request has been submitted. You will receive a confirmation soon.",
        "is_api_call": True,
        "api_endpoint": "appointment",
        "service_type": user_appointment.get("service_type", ""),
        "preferred_date": user_appointment.get("preferred_date", ""),
        "preferred_time": user_appointment.get("preferred_time", ""),
        "special_requirement": user_appointment.get("special_requirement", ""),
        "special_requirement_other": user_appointment.get("special_requirement_other", "")
    }
    
    user_appointment.clear()  

    return final_response