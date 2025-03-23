from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from openai import OpenAI
import tensorflow as tf
from PIL import Image, ImageOps
import os

app = Flask(__name__)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Configure TensorFlow (CPU usage since no GPU setup is specified)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

# Load the SavedModel
try:
    model = tf.saved_model.load("models/content")
    infer = model.signatures["serving_default"]
    print("Model input signature:", infer.structured_input_signature)
    print("Model output signature:", infer.structured_outputs)
except Exception as e:
    print(f"Error loading SavedModel: {str(e)}")
    model = None
    infer = None

# Load the labels
try:
    class_names = open("models/labels.txt", "r").readlines()
except Exception as e:
    print(f"Error loading labels: {str(e)}")
    class_names = []

# Your JSON dataset
data = [
    {
        "context": "Guideline: For deep cuts with heavy bleeding, apply pressure to control bleeding, assess for sutures, and administer tetanus prophylaxis if not up-to-date.",
        "input": "Nurse Observation: Patient has heavy bleeding and a deep cut, Image Analysis: Detected: wound (Confidence: 0.92), blood (Confidence: 0.87)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has heavy bleeding and a deep cut\n- Visual Findings: Detected: wound (Confidence: 0.92), blood (Confidence: 0.87)\n- Summary: The patient has a severe deep cut with significant bleeding, requiring immediate intervention.\n\nDoctor Prompts\n- Apply pressure to the wound to control bleeding.\n- Assess the need for sutures based on wound depth.\n- Administer tetanus prophylaxis if not up-to-date."
    },
    {
        "context": "Guideline: For patients with shortness of breath and pale skin, check vital signs, administer oxygen, and consider IV fluids if hypotensive.",
        "input": "Nurse Observation: Patient is dizzy with shallow breathing, Image Analysis: Detected: pale skin (Confidence: 0.85)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is dizzy with shallow breathing\n- Visual Findings: Detected: pale skin (Confidence: 0.85)\n- Summary: The patient shows signs of potential hypoperfusion or shock, supported by pallor observed in the image.\n\nDoctor Prompts\n- Check vital signs immediately, focusing on pulse and blood pressure.\n- Administer oxygen if respiratory distress persists.\n- Consider IV fluids if hypotension is confirmed."
    },
    {
        "context": "Guideline: For chest pain with no visible abnormalities, perform an ECG, order a chest X-ray, and monitor oxygen saturation.",
        "input": "Nurse Observation: Patient reports chest pain and shortness of breath, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient reports chest pain and shortness of breath\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient is experiencing chest pain and shortness of breath, with no visible external abnormalities, suggesting a possible internal issue such as a cardiac or respiratory condition.\n\nDoctor Prompts\n- Perform an ECG to rule out cardiac issues.\n- Order a chest X-ray to check for pulmonary conditions.\n- Monitor oxygen saturation and consider supplemental oxygen."
    },
    {
        "context": "Guideline: For swollen ankles after a fall, apply ice, elevate the ankle, and order an X-ray to rule out a fracture.",
        "input": "Nurse Observation: Patient has a swollen ankle after a fall, Image Analysis: Detected: swelling (Confidence: 0.90)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a swollen ankle after a fall\n- Visual Findings: Detected: swelling (Confidence: 0.90)\n- Summary: The patient has a swollen ankle following a fall, with imaging confirming swelling, indicating a possible sprain or fracture.\n\nDoctor Prompts\n- Apply ice and elevate the ankle to reduce swelling.\n- Order an X-ray to rule out a fracture.\n- Consider an ankle brace or immobilization if a sprain is suspected."
    },
    {
        "context": "Guideline: For unresponsive patients not breathing, initiate CPR, call for a code team, and secure an airway.",
        "input": "Nurse Observation: Patient is unresponsive and not breathing, Image Analysis: Detected: no visible movement (Confidence: 0.95)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is unresponsive and not breathing\n- Visual Findings: Detected: no visible movement (Confidence: 0.95)\n- Summary: The patient is in critical condition, unresponsive and not breathing, with no visible movement, indicating a possible cardiac arrest.\n\nDoctor Prompts\n- Initiate CPR immediately.\n- Call for a code team and prepare a defibrillator.\n- Secure an airway and administer oxygen."
    },
    {
        "context": "Guideline: For patients with rash and fever, take a history for allergens, order blood tests, and consider antihistamines or steroids.",
        "input": "Nurse Observation: Patient has a rash and fever, Image Analysis: Detected: rash (Confidence: 0.88)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a rash and fever\n- Visual Findings: Detected: rash (Confidence: 0.88)\n- Summary: The patient presents with a rash and fever, with imaging confirming the rash, suggesting a possible infectious or allergic reaction.\n\nDoctor Prompts\n- Take a detailed history to identify potential allergens or infections.\n- Order blood tests to check for infection markers.\n- Consider antihistamines or steroids if an allergic reaction is suspected."
    },
    {
        "context": "Guideline: For severe abdominal pain with distended abdomen, order an abdominal CT scan, check for peritonitis, and prepare for possible surgery.",
        "input": "Nurse Observation: Patient reports severe abdominal pain, Image Analysis: Detected: distended abdomen (Confidence: 0.91)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient reports severe abdominal pain\n- Visual Findings: Detected: distended abdomen (Confidence: 0.91)\n- Summary: The patient is experiencing severe abdominal pain with a distended abdomen, indicating a possible gastrointestinal issue such as obstruction or perforation.\n\nDoctor Prompts\n- Order an abdominal CT scan to identify the cause of distension.\n- Check for signs of peritonitis through physical examination.\n- Prepare for possible surgical intervention if a perforation is suspected."
    },
    {
        "context": "Guideline: For burns on the arm, cool the burn under running water, assess burn depth, and apply a sterile dressing.",
        "input": "Nurse Observation: Patient has a burn on the arm, Image Analysis: Detected: burn (Confidence: 0.93)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a burn on the arm\n- Visual Findings: Detected: burn (Confidence: 0.93)\n- Summary: The patient has a burn on the arm, confirmed by imaging, indicating a thermal injury that may require specialized care.\n\nDoctor Prompts\n- Cool the burn under running water for at least 10 minutes.\n- Assess the burn depth and percentage of body surface area affected.\n- Apply a sterile dressing and consider referral to a burn unit if severe."
    },
    {
        "context": "Guideline: For coughing with high fever, order a chest X-ray, collect a sputum sample, and start empiric antibiotics if bacterial infection is suspected.",
        "input": "Nurse Observation: Patient is coughing with a high fever, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is coughing with a high fever\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient presents with a cough and high fever, with no visible external abnormalities, suggesting a possible respiratory infection.\n\nDoctor Prompts\n- Order a chest X-ray to check for pneumonia.\n- Collect a sputum sample for culture and sensitivity.\n- Start empiric antibiotics if bacterial infection is suspected."
    },
    {
        "context": "Guideline: For lacerations on the forehead, clean the wound, assess for head injury, and suture if deep or gaping.",
        "input": "Nurse Observation: Patient has a laceration on the forehead, Image Analysis: Detected: laceration (Confidence: 0.89), blood (Confidence: 0.85)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a laceration on the forehead\n- Visual Findings: Detected: laceration (Confidence: 0.89), blood (Confidence: 0.85)\n- Summary: The patient has a forehead laceration with associated bleeding, confirmed by imaging, indicating a need for wound management.\n\nDoctor Prompts\n- Clean the wound thoroughly to prevent infection.\n- Assess for signs of concussion or head injury.\n- Suture the laceration if it is deep or gaping."
    },
    {
        "context": "Guideline: For seizures, protect from injury, time the seizure, and call emergency services if it lasts longer than 5 minutes or if it's the first seizure.",
        "input": "Nurse Observation: Patient is experiencing a seizure, Image Analysis: Detected: convulsions (Confidence: 0.90)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is experiencing a seizure\n- Visual Findings: Detected: convulsions (Confidence: 0.90)\n- Summary: The patient is having a seizure, confirmed by imaging, requiring immediate attention to ensure safety and assess the need for further intervention.\n\nDoctor Prompts\n- Protect the patient from injury during the seizure.\n- Time the seizure to monitor its duration.\n- Call emergency services if it lasts longer than 5 minutes or if it's the first seizure."
    },
    {
        "context": "Guideline: For stroke, use the FAST test (Face, Arm, Speech, Time), call emergency services immediately, and keep the patient calm and still.",
        "input": "Nurse Observation: Patient shows signs of stroke such as facial drooping and slurred speech, Image Analysis: Detected: facial drooping (Confidence: 0.85)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient shows signs of stroke such as facial drooping and slurred speech\n- Visual Findings: Detected: facial drooping (Confidence: 0.85)\n- Summary: The patient exhibits symptoms consistent with a stroke, confirmed by imaging, necessitating urgent medical intervention.\n\nDoctor Prompts\n- Use the FAST test (Face, Arm, Speech, Time) to confirm stroke signs.\n- Call emergency services immediately.\n- Keep the patient calm and still."
    },
    {
        "context": "Guideline: For head injuries, monitor for worsening symptoms, seek medical help if necessary, and assume neck injury.",
        "input": "Nurse Observation: Patient has a head injury after a fall, Image Analysis: Detected: bruising (Confidence: 0.87)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a head injury after a fall\n- Visual Findings: Detected: bruising (Confidence: 0.87)\n- Summary: The patient has a head injury with visible bruising, suggesting possible traumatic brain or neck injury requiring close monitoring.\n\nDoctor Prompts\n- Monitor for worsening symptoms such as confusion or vomiting.\n- Seek medical help if symptoms deteriorate.\n- Assume a neck injury and immobilize the spine."
    },
    {
        "context": "Guideline: For diabetes (hyperglycemia), dial emergency services, monitor breathing and response, and perform CPR if not breathing normally.",
        "input": "Nurse Observation: Patient with known diabetes is confused and sweating, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient with known diabetes is confused and sweating\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient with diabetes shows signs of hyperglycemia, with no visible external abnormalities, indicating a metabolic emergency.\n\nDoctor Prompts\n- Dial emergency services for immediate assistance.\n- Monitor breathing and response closely.\n- Perform CPR if the patient stops breathing normally."
    },
    {
        "context": "Guideline: For diabetes (hypoglycemia), if responsive, help take 15-20g of glucose; if unresponsive, place in recovery position and call emergency services.",
        "input": "Nurse Observation: Patient with diabetes is shaky and unresponsive, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient with diabetes is shaky and unresponsive\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient with diabetes is unresponsive, likely due to hypoglycemia, requiring urgent intervention despite no visible external signs.\n\nDoctor Prompts\n- If responsive, help the patient take 15-20g of glucose.\n- If unresponsive, place in the recovery position.\n- Call emergency services immediately."
    },
    {
        "context": "Guideline: For choking (child/adult), encourage coughing, perform 5 back blows, then 5 abdominal thrusts, and call emergency services if the object is not dislodged.",
        "input": "Nurse Observation: Patient is choking and unable to speak, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is choking and unable to speak\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient is experiencing a choking emergency, with no visible external signs, requiring immediate airway clearance.\n\nDoctor Prompts\n- Encourage coughing to dislodge the object.\n- Perform 5 back blows followed by 5 abdominal thrusts.\n- Call emergency services if the object is not dislodged."
    },
    {
        "context": "Guideline: For choking (baby), perform 5 back blows on the thigh, then 5 chest thrusts, and call emergency services if the object is not dislodged.",
        "input": "Nurse Observation: Baby is choking and turning blue, Image Analysis: Detected: cyanosis (Confidence: 0.89)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Baby is choking and turning blue\n- Visual Findings: Detected: cyanosis (Confidence: 0.89)\n- Summary: The baby is choking with visible cyanosis, indicating a critical airway obstruction requiring immediate intervention.\n\nDoctor Prompts\n- Perform 5 back blows on the thigh.\n- Perform 5 chest thrusts if the object remains.\n- Call emergency services if the object is not dislodged."
    },
    {
        "context": "Guideline: For asthma, use a reliever inhaler (1-2 puffs every 30-60 seconds up to 10 puffs), and call emergency services if symptoms worsen or the patient is exhausted.",
        "input": "Nurse Observation: Patient with asthma is wheezing and short of breath, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient with asthma is wheezing and short of breath\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient with asthma is experiencing an acute attack, with no visible external signs, requiring immediate respiratory support.\n\nDoctor Prompts\n- Use a reliever inhaler (1-2 puffs every 30-60 seconds up to 10 puffs).\n- Call emergency services if symptoms worsen or the patient becomes exhausted.\n- Monitor breathing and oxygen levels."
    },
    {
        "context": "Guideline: For minor allergic reactions, assess symptoms, treat as needed, and advise seeking medical advice.",
        "input": "Nurse Observation: Patient has mild itching and a slight rash, Image Analysis: Detected: rash (Confidence: 0.84)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has mild itching and a slight rash\n- Visual Findings: Detected: rash (Confidence: 0.84)\n- Summary: The patient is experiencing a minor allergic reaction, confirmed by a visible rash, requiring symptomatic treatment.\n\nDoctor Prompts\n- Assess symptoms to determine the severity of the reaction.\n- Treat with antihistamines or topical creams as needed.\n- Advise the patient to seek medical advice if symptoms persist."
    },
    {
        "context": "Guideline: For severe allergic reactions (anaphylaxis), dial emergency services, help use an auto-injector, and monitor for shock.",
        "input": "Nurse Observation: Patient has swelling and difficulty breathing, Image Analysis: Detected: swelling (Confidence: 0.91)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has swelling and difficulty breathing\n- Visual Findings: Detected: swelling (Confidence: 0.91)\n- Summary: The patient is experiencing a severe allergic reaction (anaphylaxis), with visible swelling, indicating a life-threatening emergency.\n\nDoctor Prompts\n- Dial emergency services immediately.\n- Help the patient use an auto-injector (e.g., epinephrine) if available.\n- Monitor for signs of shock and support breathing."
    },
    {
        "context": "Guideline: For chest pains (heart attack), dial emergency services, give 300mg aspirin if no contraindications, and monitor vital signs.",
        "input": "Nurse Observation: Patient reports severe chest pain radiating to the arm, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient reports severe chest pain radiating to the arm\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient is experiencing severe chest pain suggestive of a heart attack, with no visible external signs, requiring urgent cardiac care.\n\nDoctor Prompts\n- Dial emergency services immediately.\n- Give 300mg aspirin if no contraindications are present.\n- Monitor vital signs, especially heart rate and blood pressure."
    },
    {
        "context": "Guideline: For shock, lay the patient down, raise their legs, and call emergency services while monitoring vital signs.",
        "input": "Nurse Observation: Patient is pale and clammy with rapid pulse, Image Analysis: Detected: pale skin (Confidence: 0.86)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is pale and clammy with rapid pulse\n- Visual Findings: Detected: pale skin (Confidence: 0.86)\n- Summary: The patient is in shock, with visible pallor confirming circulatory distress, requiring immediate stabilization.\n\nDoctor Prompts\n- Lay the patient down and raise their legs to improve circulation.\n- Call emergency services for urgent assistance.\n- Monitor vital signs continuously."
    },
    {
        "context": "Guideline: For fainting, lay the patient down, raise their legs, and call emergency services if they remain unresponsive.",
        "input": "Nurse Observation: Patient fainted and is now awake but weak, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient fainted and is now awake but weak\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient experienced a fainting episode and remains weak, with no visible external signs, suggesting a need for further evaluation.\n\nDoctor Prompts\n- Lay the patient down and raise their legs to improve blood flow.\n- Call emergency services if the patient remains unresponsive.\n- Monitor for recurrence or worsening symptoms."
    },
    {
        "context": "Guideline: For minor bleeding, clean the wound, cover with a sterile dressing, and elevate the injured area.",
        "input": "Nurse Observation: Patient has a small cut with mild bleeding, Image Analysis: Detected: blood (Confidence: 0.83)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a small cut with mild bleeding\n- Visual Findings: Detected: blood (Confidence: 0.83)\n- Summary: The patient has a minor cut with mild bleeding, confirmed by imaging, requiring basic wound care.\n\nDoctor Prompts\n- Clean the wound to prevent infection.\n- Cover with a sterile dressing.\n- Elevate the injured area to reduce bleeding."
    },
    {
        "context": "Guideline: For severe bleeding, apply direct pressure, use a dressing, and call emergency services while treating for shock.",
        "input": "Nurse Observation: Patient has severe bleeding from a leg wound, Image Analysis: Detected: blood (Confidence: 0.94)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has severe bleeding from a leg wound\n- Visual Findings: Detected: blood (Confidence: 0.94)\n- Summary: The patient is experiencing severe bleeding from a leg wound, confirmed by imaging, indicating a critical hemorrhage.\n\nDoctor Prompts\n- Apply direct pressure to control bleeding.\n- Use a dressing to cover the wound.\n- Call emergency services and treat for shock."
    },
    {
        "context": "Guideline: For catastrophic bleeding, use a haemostatic dressing or tourniquet, and call emergency services immediately.",
        "input": "Nurse Observation: Patient has uncontrollable bleeding from an arm injury, Image Analysis: Detected: blood (Confidence: 0.96)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has uncontrollable bleeding from an arm injury\n- Visual Findings: Detected: blood (Confidence: 0.96)\n- Summary: The patient has catastrophic bleeding from an arm injury, with significant blood loss confirmed by imaging, requiring urgent intervention.\n\nDoctor Prompts\n- Use a haemostatic dressing or tourniquet to stop bleeding.\n- Call emergency services immediately.\n- Monitor for signs of shock."
    },
    {
        "context": "Guideline: For amputation, control bleeding, wrap the severed part in a plastic bag with ice, and call emergency services.",
        "input": "Nurse Observation: Patient has a severed finger with heavy bleeding, Image Analysis: Detected: blood (Confidence: 0.95), missing digit (Confidence: 0.92)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a severed finger with heavy bleeding\n- Visual Findings: Detected: blood (Confidence: 0.95), missing digit (Confidence: 0.92)\n- Summary: The patient has suffered an amputation with significant bleeding, confirmed by imaging, requiring immediate care and preservation of the severed part.\n\nDoctor Prompts\n- Control bleeding with direct pressure or a tourniquet.\n- Wrap the severed part in a plastic bag with ice.\n- Call emergency services for urgent transport."
    },
    {
        "context": "Guideline: For superficial burns, cool under water for 20 minutes, cover with a sterile dressing, and seek medical help for severe cases.",
        "input": "Nurse Observation: Patient has a superficial burn on the hand, Image Analysis: Detected: burn (Confidence: 0.90)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a superficial burn on the hand\n- Visual Findings: Detected: burn (Confidence: 0.90)\n- Summary: The patient has a superficial burn on the hand, confirmed by imaging, indicating a need for basic burn care.\n\nDoctor Prompts\n- Cool the burn under water for 20 minutes.\n- Cover with a sterile dressing.\n- Seek medical help if the burn is severe or extensive."
    },
    {
        "context": "Guideline: For chemical burns, flush with water for at least 20 minutes and call emergency services.",
        "input": "Nurse Observation: Patient has a chemical burn on the leg, Image Analysis: Detected: burn (Confidence: 0.91)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a chemical burn on the leg\n- Visual Findings: Detected: burn (Confidence: 0.91)\n- Summary: The patient has a chemical burn on the leg, confirmed by imaging, requiring immediate decontamination and medical attention.\n\nDoctor Prompts\n- Flush the burn with water for at least 20 minutes.\n- Call emergency services for further management.\n- Monitor for systemic effects of chemical exposure."
    },
    {
        "context": "Guideline: For smoke inhalation, move to fresh air, administer oxygen if available, and monitor for respiratory distress.",
        "input": "Nurse Observation: Patient reports difficulty breathing after smoke exposure, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient reports difficulty breathing after smoke exposure\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has likely suffered smoke inhalation, with no visible external signs, indicating a need for respiratory support.\n\nDoctor Prompts\n- Move the patient to fresh air immediately.\n- Administer oxygen if available.\n- Monitor for respiratory distress and escalate care if needed."
    },
    {
        "context": "Guideline: For chemical burns to the eye, irrigate with cool water for at least 20 minutes, and call emergency services.",
        "input": "Nurse Observation: Patient reports eye pain after chemical splash, Image Analysis: Detected: redness (Confidence: 0.88)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient reports eye pain after chemical splash\n- Visual Findings: Detected: redness (Confidence: 0.88)\n- Summary: The patient has a chemical burn to the eye, with visible redness, indicating a need for urgent irrigation and care.\n\nDoctor Prompts\n- Irrigate the eye with cool water for at least 20 minutes.\n- Call emergency services for immediate assistance.\n- Avoid rubbing the eye and protect it from further damage."
    },
    {
        "context": "Guideline: For swallowed poisons, identify the substance, call poison control, and monitor vital signs while preparing for CPR if unresponsive.",
        "input": "Nurse Observation: Patient ingested a toxic substance and is vomiting, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient ingested a toxic substance and is vomiting\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has swallowed a poison, with no visible external signs, requiring immediate poison management.\n\nDoctor Prompts\n- Identify the substance ingested.\n- Call poison control for specific guidance.\n- Monitor vital signs and prepare for CPR if the patient becomes unresponsive."
    },
    {
        "context": "Guideline: For skin contact with poisons, flush with water for at least 20 minutes, and call emergency services if necessary.",
        "input": "Nurse Observation: Patient has skin irritation after chemical contact, Image Analysis: Detected: redness (Confidence: 0.87)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has skin irritation after chemical contact\n- Visual Findings: Detected: redness (Confidence: 0.87)\n- Summary: The patient has skin exposure to a poison, with visible redness, requiring decontamination.\n\nDoctor Prompts\n- Flush the skin with water for at least 20 minutes.\n- Call emergency services if symptoms worsen or systemic effects occur.\n- Monitor for additional reactions."
    },
    {
        "context": "Guideline: For poisonous gases, move to fresh air, perform CPR if not breathing normally, and call emergency services.",
        "input": "Nurse Observation: Patient is dizzy after gas exposure, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is dizzy after gas exposure\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has been exposed to poisonous gases, with no visible signs, indicating a need for respiratory support.\n\nDoctor Prompts\n- Move the patient to fresh air immediately.\n- Perform CPR if the patient is not breathing normally.\n- Call emergency services for urgent care."
    },
    {
        "context": "Guideline: For foreign objects in the eye, do not rub, rinse with clean water, and seek medical help if not removed.",
        "input": "Nurse Observation: Patient reports eye irritation from a foreign object, Image Analysis: Detected: redness (Confidence: 0.85)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient reports eye irritation from a foreign object\n- Visual Findings: Detected: redness (Confidence: 0.85)\n- Summary: The patient has a foreign object in the eye, with visible redness, requiring careful removal.\n\nDoctor Prompts\n- Do not rub the eye to avoid further damage.\n- Rinse with clean water to attempt removal.\n- Seek medical help if the object is not removed."
    },
    {
        "context": "Guideline: For foreign objects in the ear, do not remove lodged objects, flood with tepid water for insects, and seek medical help if unsuccessful.",
        "input": "Nurse Observation: Patient reports ear discomfort from a foreign object, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient reports ear discomfort from a foreign object\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has a foreign object in the ear, with no visible external signs, requiring careful management.\n\nDoctor Prompts\n- Do not attempt to remove lodged objects manually.\n- Flood the ear with tepid water if an insect is suspected.\n- Seek medical help if unsuccessful."
    },
    {
        "context": "Guideline: For foreign objects in the nose, keep calm, breathe through mouth, and send to hospital if lodged.",
        "input": "Nurse Observation: Patient has difficulty breathing through nose due to a foreign object, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has difficulty breathing through nose due to a foreign object\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has a foreign object in the nose, with no visible external signs, requiring medical attention if lodged.\n\nDoctor Prompts\n- Keep the patient calm to prevent further distress.\n- Instruct to breathe through the mouth.\n- Send to hospital if the object remains lodged."
    },
    {
        "context": "Guideline: For hypothermia, move to shelter, remove wet clothing, provide dry clothing/blankets, and call emergency services.",
        "input": "Nurse Observation: Patient is shivering and cold after exposure, Image Analysis: Detected: pale skin (Confidence: 0.88)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is shivering and cold after exposure\n- Visual Findings: Detected: pale skin (Confidence: 0.88)\n- Summary: The patient is experiencing hypothermia, with visible pallor, indicating a need for warming and medical care.\n\nDoctor Prompts\n- Move the patient to shelter to prevent further heat loss.\n- Remove wet clothing and provide dry clothing or blankets.\n- Call emergency services for further management."
    },
    {
        "context": "Guideline: For heat exhaustion, move to a cool place, lay down, raise legs, encourage water, and call emergency services if worsens.",
        "input": "Nurse Observation: Patient is dizzy and sweating heavily after heat exposure, Image Analysis: Detected: sweating (Confidence: 0.90)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is dizzy and sweating heavily after heat exposure\n- Visual Findings: Detected: sweating (Confidence: 0.90)\n- Summary: The patient is experiencing heat exhaustion, with visible sweating, requiring cooling and hydration.\n\nDoctor Prompts\n- Move the patient to a cool place.\n- Lay the patient down and raise their legs.\n- Encourage water intake and call emergency services if symptoms worsen."
    },
    {
        "context": "Guideline: For heatstroke, move to a cool place, remove outer clothing, wrap in cold wet sheet, and call emergency services.",
        "input": "Nurse Observation: Patient is confused and hot after heat exposure, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is confused and hot after heat exposure\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient is experiencing heatstroke, with no visible external signs, indicating a life-threatening condition requiring urgent cooling.\n\nDoctor Prompts\n- Move the patient to a cool place.\n- Remove outer clothing and wrap in a cold wet sheet.\n- Call emergency services immediately."
    },
    {
        "context": "Guideline: For sunburn, move out of sun, cover skin, cool with water for 10 minutes, and seek medical help for severe cases.",
        "input": "Nurse Observation: Patient has red and painful skin after sun exposure, Image Analysis: Detected: redness (Confidence: 0.89)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has red and painful skin after sun exposure\n- Visual Findings: Detected: redness (Confidence: 0.89)\n- Summary: The patient has sunburn, with visible redness, requiring cooling and protection.\n\nDoctor Prompts\n- Move the patient out of the sun.\n- Cover the skin to prevent further damage.\n- Cool with water for 10 minutes and seek medical help if severe."
    },
    {
        "context": "Guideline: For meningitis, dial emergency services for urgent attention, and monitor condition until help arrives.",
        "input": "Nurse Observation: Patient has a stiff neck and high fever, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a stiff neck and high fever\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient shows signs of meningitis, with no visible external signs, indicating a need for urgent medical attention.\n\nDoctor Prompts\n- Dial emergency services for urgent attention.\n- Monitor the patient’s condition until help arrives.\n- Prepare for possible lumbar puncture or antibiotics."
    },
    {
        "context": "Guideline: For sepsis, call emergency services, ask if it could be sepsis, monitor, and keep cool until help arrives.",
        "input": "Nurse Observation: Patient has a fever and rapid breathing, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a fever and rapid breathing\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient may have sepsis, with no visible external signs, requiring immediate medical evaluation.\n\nDoctor Prompts\n- Call emergency services immediately.\n- Ask if it could be sepsis based on symptoms.\n- Monitor vital signs and keep the patient cool until help arrives."
    },
    {
        "context": "Guideline: For animal/human bites, wash with soap and water, cover with a sterile dressing, and seek hospital care for large/deep wounds.",
        "input": "Nurse Observation: Patient has a bite wound on the hand, Image Analysis: Detected: wound (Confidence: 0.90)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a bite wound on the hand\n- Visual Findings: Detected: wound (Confidence: 0.90)\n- Summary: The patient has an animal or human bite wound, confirmed by imaging, requiring cleaning and potential hospital care.\n\nDoctor Prompts\n- Wash the wound with soap and water.\n- Cover with a sterile dressing.\n- Seek hospital care for large or deep wounds."
    },
    {
        "context": "Guideline: For stings, remove the sting, apply a cold compress for 20 minutes, and seek medical advice if symptoms persist.",
        "input": "Nurse Observation: Patient has pain and swelling from a sting, Image Analysis: Detected: swelling (Confidence: 0.87)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has pain and swelling from a sting\n- Visual Findings: Detected: swelling (Confidence: 0.87)\n- Summary: The patient has a sting with visible swelling, indicating a need for symptomatic relief.\n\nDoctor Prompts\n- Remove the sting if visible.\n- Apply a cold compress for 20 minutes.\n- Seek medical advice if symptoms persist or worsen."
    },
    {
        "context": "Guideline: For tick bites, use tweezers for removal, clean the area, and advise seeking medical advice.",
        "input": "Nurse Observation: Patient has a tick attached to the skin, Image Analysis: Detected: tick (Confidence: 0.92)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a tick attached to the skin\n- Visual Findings: Detected: tick (Confidence: 0.92)\n- Summary: The patient has a tick bite, confirmed by imaging, requiring removal and monitoring for infection.\n\nDoctor Prompts\n- Use tweezers to remove the tick carefully.\n- Clean the area thoroughly.\n- Advise seeking medical advice for signs of infection."
    },
    {
        "context": "Guideline: For sprains/strains, use RICE procedure (Rest, Ice, Compression, Elevation), and seek medical help if severe.",
        "input": "Nurse Observation: Patient has ankle pain and swelling after twisting it, Image Analysis: Detected: swelling (Confidence: 0.89)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has ankle pain and swelling after twisting it\n- Visual Findings: Detected: swelling (Confidence: 0.89)\n- Summary: The patient has a sprain or strain with visible swelling, requiring the RICE procedure.\n\nDoctor Prompts\n- Use the RICE procedure (Rest, Ice, Compression, Elevation).\n- Seek medical help if the injury is severe or doesn’t improve.\n- Monitor for signs of a fracture."
    },
    {
        "context": "Guideline: For fractures, immobilize the limb, apply ice, and transport to hospital for evaluation.",
        "input": "Nurse Observation: Patient has severe pain and deformity in the arm after a fall, Image Analysis: Detected: deformity (Confidence: 0.93)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has severe pain and deformity in the arm after a fall\n- Visual Findings: Detected: deformity (Confidence: 0.93)\n- Summary: The patient has a likely fracture with visible deformity, requiring immobilization and hospital care.\n\nDoctor Prompts\n- Immobilize the limb to prevent further injury.\n- Apply ice to reduce swelling.\n- Transport to hospital for evaluation and imaging."
    },
    {
        "context": "Guideline: For dislocations, immobilize, provide pain relief, and transport for reduction.",
        "input": "Nurse Observation: Patient has a shoulder out of place after a fall, Image Analysis: Detected: deformity (Confidence: 0.91)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a shoulder out of place after a fall\n- Visual Findings: Detected: deformity (Confidence: 0.91)\n- Summary: The patient has a dislocation with visible deformity, requiring immobilization and medical reduction.\n\nDoctor Prompts\n- Immobilize the joint to prevent further damage.\n- Provide pain relief as needed.\n- Transport to a facility for reduction."
    },
    {
        "context": "Guideline: For spinal injuries, immobilize the spine, call emergency services, and do not move unless necessary.",
        "input": "Nurse Observation: Patient reports neck pain after a car accident, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient reports neck pain after a car accident\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient may have a spinal injury, with no visible external signs, requiring careful immobilization.\n\nDoctor Prompts\n- Immobilize the spine to prevent further injury.\n- Call emergency services immediately.\n- Do not move the patient unless in immediate danger."
    },
    {
        "context": "Guideline: For eye injuries, protect the eye, do not apply pressure, and seek immediate medical attention.",
        "input": "Nurse Observation: Patient has eye pain and tearing after trauma, Image Analysis: Detected: redness (Confidence: 0.90)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has eye pain and tearing after trauma\n- Visual Findings: Detected: redness (Confidence: 0.90)\n- Summary: The patient has an eye injury with visible redness, requiring urgent care to prevent vision loss.\n\nDoctor Prompts\n- Protect the eye with a shield or cover.\n- Do not apply pressure to the eye.\n- Seek immediate medical attention."
    },
    {
        "context": "Guideline: For ear injuries, avoid cleaning inside the ear, and seek medical help for foreign objects or trauma.",
        "input": "Nurse Observation: Patient reports ear pain after an object insertion, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient reports ear pain after an object insertion\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has an ear injury, with no visible external signs, requiring professional evaluation.\n\nDoctor Prompts\n- Avoid cleaning inside the ear to prevent further damage.\n- Seek medical help for foreign objects or trauma.\n- Monitor for hearing changes or bleeding."
    },
    {
        "context": "Guideline: For nose injuries, control bleeding by pinching, apply cold compress, and seek help for suspected fractures.",
        "input": "Nurse Observation: Patient has a nosebleed after a hit, Image Analysis: Detected: blood (Confidence: 0.88)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a nosebleed after a hit\n- Visual Findings: Detected: blood (Confidence: 0.88)\n- Summary: The patient has a nose injury with visible bleeding, requiring control and possible fracture evaluation.\n\nDoctor Prompts\n- Control bleeding by pinching the nose.\n- Apply a cold compress to reduce swelling.\n- Seek help if a fracture is suspected."
    },
    {
        "context": "Guideline: For dental emergencies, for knocked-out teeth, preserve in milk, and seek dental care immediately.",
        "input": "Nurse Observation: Patient has a knocked-out tooth after trauma, Image Analysis: Detected: missing tooth (Confidence: 0.94)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a knocked-out tooth after trauma\n- Visual Findings: Detected: missing tooth (Confidence: 0.94)\n- Summary: The patient has a dental emergency with a knocked-out tooth, confirmed by imaging, requiring preservation and urgent care.\n\nDoctor Prompts\n- Preserve the knocked-out tooth in milk.\n- Seek dental care immediately.\n- Avoid handling the tooth root."
    },
    {
        "context": "Guideline: For marine animal stings, remove spines or tentacles, immerse in hot water, and seek medical help.",
        "input": "Nurse Observation: Patient has pain and swelling from a marine sting, Image Analysis: Detected: swelling (Confidence: 0.90)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has pain and swelling from a marine sting\n- Visual Findings: Detected: swelling (Confidence: 0.90)\n- Summary: The patient has a marine animal sting with visible swelling, requiring removal of stingers and pain relief.\n\nDoctor Prompts\n- Remove spines or tentacles carefully.\n- Immerse the area in hot water for pain relief.\n- Seek medical help for severe reactions."
    },
    {
        "context": "Guideline: For snake bites, immobilize the bitten limb, keep at heart level, and call emergency services.",
        "input": "Nurse Observation: Patient has a snake bite on the leg, Image Analysis: Detected: puncture marks (Confidence: 0.92)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a snake bite on the leg\n- Visual Findings: Detected: puncture marks (Confidence: 0.92)\n- Summary: The patient has a snake bite with visible puncture marks, indicating a potential envenomation requiring urgent care.\n\nDoctor Prompts\n- Immobilize the bitten limb.\n- Keep the limb at heart level.\n- Call emergency services immediately."
    },
    {
        "context": "Guideline: For spider bites, clean the area, apply cold compress, and seek medical help for black widow or brown recluse bites.",
        "input": "Nurse Observation: Patient has a spider bite with redness, Image Analysis: Detected: redness (Confidence: 0.88)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a spider bite with redness\n- Visual Findings: Detected: redness (Confidence: 0.88)\n- Summary: The patient has a spider bite with visible redness, requiring cleaning and monitoring for severe species.\n\nDoctor Prompts\n- Clean the area to prevent infection.\n- Apply a cold compress for swelling.\n- Seek medical help for black widow or brown recluse bites."
    },
    {
        "context": "Guideline: For tick removal, use tweezers to remove, clean the area, and monitor for Lyme disease symptoms.",
        "input": "Nurse Observation: Patient has a tick embedded in the skin, Image Analysis: Detected: tick (Confidence: 0.93)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a tick embedded in the skin\n- Visual Findings: Detected: tick (Confidence: 0.93)\n- Summary: The patient has a tick bite, confirmed by imaging, requiring removal and monitoring for infection.\n\nDoctor Prompts\n- Use tweezers to remove the tick carefully.\n- Clean the area thoroughly.\n- Monitor for Lyme disease symptoms."
    },
    {
        "context": "Guideline: For jellyfish stings, rinse with vinegar or seawater, remove tentacles, and apply hot water or heat pack.",
        "input": "Nurse Observation: Patient has pain from a jellyfish sting, Image Analysis: Detected: redness (Confidence: 0.89)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has pain from a jellyfish sting\n- Visual Findings: Detected: redness (Confidence: 0.89)\n- Summary: The patient has a jellyfish sting with visible redness, requiring specific treatment to alleviate pain.\n\nDoctor Prompts\n- Rinse with vinegar or seawater to neutralize venom.\n- Remove tentacles carefully.\n- Apply hot water or a heat pack for pain relief."
    },
    {
        "context": "Guideline: For bee stings, remove stinger, apply cold compress, and monitor for allergic reactions.",
        "input": "Nurse Observation: Patient has a bee sting with swelling, Image Analysis: Detected: swelling (Confidence: 0.87)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a bee sting with swelling\n- Visual Findings: Detected: swelling (Confidence: 0.87)\n- Summary: The patient has a bee sting with visible swelling, requiring stinger removal and monitoring.\n\nDoctor Prompts\n- Remove the stinger if present.\n- Apply a cold compress to reduce swelling.\n- Monitor for allergic reactions."
    },
    {
        "context": "Guideline: For wasp stings, apply cold compress, and monitor for allergic reactions.",
        "input": "Nurse Observation: Patient has pain from a wasp sting, Image Analysis: Detected: redness (Confidence: 0.86)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has pain from a wasp sting\n- Visual Findings: Detected: redness (Confidence: 0.86)\n- Summary: The patient has a wasp sting with visible redness, requiring symptomatic care and observation.\n\nDoctor Prompts\n- Apply a cold compress to reduce pain and swelling.\n- Monitor for allergic reactions.\n- Seek help if symptoms escalate."
    },
    {
        "context": "Guideline: For plant poisoning, identify the plant, call poison control, and provide supportive care.",
        "input": "Nurse Observation: Patient ingested a toxic plant and is nauseous, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient ingested a toxic plant and is nauseous\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has plant poisoning, with no visible external signs, requiring poison management.\n\nDoctor Prompts\n- Identify the plant ingested.\n- Call poison control for specific advice.\n- Provide supportive care such as hydration."
    },
    {
        "context": "Guideline: For food poisoning, hydrate, monitor for dehydration, and seek medical help if severe.",
        "input": "Nurse Observation: Patient is vomiting after eating contaminated food, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is vomiting after eating contaminated food\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has food poisoning, with no visible external signs, requiring hydration and monitoring.\n\nDoctor Prompts\n- Hydrate the patient with small sips of water.\n- Monitor for signs of dehydration.\n- Seek medical help if symptoms are severe."
    },
    {
        "context": "Guideline: For alcohol poisoning, monitor vital signs, position on side to prevent aspiration, and call emergency services.",
        "input": "Nurse Observation: Patient is unresponsive after heavy drinking, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is unresponsive after heavy drinking\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has alcohol poisoning, with no visible external signs, indicating a critical condition.\n\nDoctor Prompts\n- Monitor vital signs closely.\n- Position the patient on their side to prevent aspiration.\n- Call emergency services immediately."
    },
    {
        "context": "Guideline: For drug overdose, identify the drug, call poison control or emergency services, and provide supportive care.",
        "input": "Nurse Observation: Patient is drowsy after taking excess medication, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is drowsy after taking excess medication\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has a drug overdose, with no visible external signs, requiring urgent intervention.\n\nDoctor Prompts\n- Identify the drug taken.\n- Call poison control or emergency services.\n- Provide supportive care such as monitoring breathing."
    },
    {
        "context": "Guideline: For carbon monoxide poisoning, remove from source, administer 100% oxygen, and call emergency services.",
        "input": "Nurse Observation: Patient is confused after gas exposure, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is confused after gas exposure\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has carbon monoxide poisoning, with no visible signs, requiring immediate oxygen therapy.\n\nDoctor Prompts\n- Remove the patient from the source of exposure.\n- Administer 100% oxygen if available.\n- Call emergency services for urgent care."
    },
    {
        "context": "Guideline: For drowning, remove from water, check for breathing, and start CPR if necessary.",
        "input": "Nurse Observation: Patient was pulled from water and is unresponsive, Image Analysis: Detected: no visible movement (Confidence: 0.94)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient was pulled from water and is unresponsive\n- Visual Findings: Detected: no visible movement (Confidence: 0.94)\n- Summary: The patient has drowned and is unresponsive, with no visible movement, indicating a need for resuscitation.\n\nDoctor Prompts\n- Remove the patient from water.\n- Check for breathing and pulse.\n- Start CPR if necessary."
    },
    {
        "context": "Guideline: For near drowning, monitor for delayed pulmonary edema, and seek medical help.",
        "input": "Nurse Observation: Patient is coughing after water submersion, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is coughing after water submersion\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient experienced a near drowning, with no visible signs, requiring monitoring for complications.\n\nDoctor Prompts\n- Monitor for delayed pulmonary edema.\n- Seek medical help for evaluation.\n- Provide oxygen if respiratory distress develops."
    },
    {
        "context": "Guideline: For electrical burns, turn off power, check for exit wounds, treat as burns, and monitor for arrhythmias.",
        "input": "Nurse Observation: Patient has burns from an electrical shock, Image Analysis: Detected: burn (Confidence: 0.92)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has burns from an electrical shock\n- Visual Findings: Detected: burn (Confidence: 0.92)\n- Summary: The patient has electrical burns, confirmed by imaging, requiring specific burn care and cardiac monitoring.\n\nDoctor Prompts\n- Turn off power source before approaching.\n- Check for exit wounds from the electrical current.\n- Treat as burns and monitor for arrhythmias."
    },
    {
        "context": "Guideline: For lightning strikes, check for cardiac arrest, start CPR if necessary, and monitor for neurological deficits.",
        "input": "Nurse Observation: Patient was struck by lightning and is unresponsive, Image Analysis: Detected: no visible movement (Confidence: 0.95)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient was struck by lightning and is unresponsive\n- Visual Findings: Detected: no visible movement (Confidence: 0.95)\n- Summary: The patient was struck by lightning and is unresponsive, with no visible movement, indicating a need for resuscitation.\n\nDoctor Prompts\n- Check for cardiac arrest immediately.\n- Start CPR if necessary.\n- Monitor for neurological deficits."
    },
    {
        "context": "Guideline: For chemical exposure, remove contaminated clothing, flush skin and eyes with water, and call emergency services.",
        "input": "Nurse Observation: Patient has skin irritation after chemical spill, Image Analysis: Detected: redness (Confidence: 0.89)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has skin irritation after chemical spill\n- Visual Findings: Detected: redness (Confidence: 0.89)\n- Summary: The patient has chemical exposure with visible redness, requiring decontamination and medical attention.\n\nDoctor Prompts\n- Remove contaminated clothing carefully.\n- Flush skin and eyes with water.\n- Call emergency services for further care."
    },
    {
        "context": "Guideline: For radiation exposure, remove from source, decontaminate, and monitor for acute radiation syndrome.",
        "input": "Nurse Observation: Patient was exposed to radiation and feels nauseous, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient was exposed to radiation and feels nauseous\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has radiation exposure, with no visible signs, requiring decontamination and monitoring.\n\nDoctor Prompts\n- Remove the patient from the radiation source.\n- Decontaminate by removing clothing and washing.\n- Monitor for acute radiation syndrome symptoms."
    },
    {
        "context": "Guideline: For blast injuries, assess for primary, secondary, tertiary, and quaternary injuries, and call emergency services.",
        "input": "Nurse Observation: Patient was near an explosion and has multiple injuries, Image Analysis: Detected: wounds (Confidence: 0.91)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient was near an explosion and has multiple injuries\n- Visual Findings: Detected: wounds (Confidence: 0.91)\n- Summary: The patient has blast injuries with visible wounds, indicating a need for comprehensive assessment.\n\nDoctor Prompts\n- Assess for primary, secondary, tertiary, and quaternary injuries.\n- Call emergency services immediately.\n- Stabilize the patient as needed."
    },
    {
        "context": "Guideline: For concussions, monitor for loss of consciousness, amnesia, confusion, and rest and observe.",
        "input": "Nurse Observation: Patient is dazed after a head blow, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is dazed after a head blow\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has a possible concussion, with no visible signs, requiring monitoring and rest.\n\nDoctor Prompts\n- Monitor for loss of consciousness, amnesia, or confusion.\n- Ensure the patient rests and avoids activity.\n- Observe for worsening symptoms."
    },
    {
        "context": "Guideline: For altitude sickness, descend to lower altitude, administer oxygen, and consider acetazolamide for prevention.",
        "input": "Nurse Observation: Patient has headache and nausea at high altitude, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has headache and nausea at high altitude\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has altitude sickness, with no visible signs, requiring descent and oxygen.\n\nDoctor Prompts\n- Descend to a lower altitude immediately.\n- Administer oxygen if available.\n- Consider acetazolamide for prevention or treatment."
    },
    {
        "context": "Guideline: For decompression sickness, administer 100% oxygen, and transport to a hyperbaric chamber.",
        "input": "Nurse Observation: Patient has joint pain after diving, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has joint pain after diving\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has decompression sickness, with no visible signs, requiring oxygen and specialized care.\n\nDoctor Prompts\n- Administer 100% oxygen immediately.\n- Transport to a hyperbaric chamber.\n- Monitor for worsening symptoms."
    },
    {
        "context": "Guideline: For frostbite, warm affected areas in warm water, avoid rubbing, and seek medical help.",
        "input": "Nurse Observation: Patient has numb and discolored fingers after cold exposure, Image Analysis: Detected: discoloration (Confidence: 0.90)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has numb and discolored fingers after cold exposure\n- Visual Findings: Detected: discoloration (Confidence: 0.90)\n- Summary: The patient has frostbite with visible discoloration, requiring warming and medical care.\n\nDoctor Prompts\n- Warm affected areas in warm water (not hot).\n- Avoid rubbing the affected areas.\n- Seek medical help for severe cases."
    },
    {
        "context": "Guideline: For trench foot, dry the feet, warm gradually, elevate, and seek medical help.",
        "input": "Nurse Observation: Patient has cold, wet feet with swelling, Image Analysis: Detected: swelling (Confidence: 0.88)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has cold, wet feet with swelling\n- Visual Findings: Detected: swelling (Confidence: 0.88)\n- Summary: The patient has trench foot with visible swelling, requiring drying and warming.\n\nDoctor Prompts\n- Dry the feet thoroughly.\n- Warm the feet gradually and elevate.\n- Seek medical help for infection or severe damage."
    },
    {
        "context": "Guideline: For snow blindness, cover eyes, use dark glasses, and seek medical help.",
        "input": "Nurse Observation: Patient has eye pain and blurred vision after snow exposure, Image Analysis: Detected: redness (Confidence: 0.87)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has eye pain and blurred vision after snow exposure\n- Visual Findings: Detected: redness (Confidence: 0.87)\n- Summary: The patient has snow blindness with visible redness, requiring eye protection and care.\n\nDoctor Prompts\n- Cover the eyes to prevent further damage.\n- Use dark glasses to reduce light exposure.\n- Seek medical help for persistent symptoms."
    },
    {
        "context": "Guideline: For avalanche burial, dig out, check for airway, and start CPR if necessary.",
        "input": "Nurse Observation: Patient was buried in an avalanche and is unresponsive, Image Analysis: Detected: no visible movement (Confidence: 0.95)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient was buried in an avalanche and is unresponsive\n- Visual Findings: Detected: no visible movement (Confidence: 0.95)\n- Summary: The patient was buried in an avalanche and is unresponsive, with no visible movement, requiring immediate rescue and resuscitation.\n\nDoctor Prompts\n- Dig the patient out quickly.\n- Check for airway and breathing.\n- Start CPR if necessary."
    },
    {
        "context": "Guideline: For hypoglycemia in diabetics, administer glucose, monitor blood sugar, and call emergency services if unconscious.",
        "input": "Nurse Observation: Diabetic patient is sweating and confused, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Diabetic patient is sweating and confused\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The diabetic patient has hypoglycemia, with no visible signs, requiring glucose administration.\n\nDoctor Prompts\n- Administer glucose (15-20g) if conscious.\n- Monitor blood sugar levels.\n- Call emergency services if the patient becomes unconscious."
    },
    {
        "context": "Guideline: For hyperglycemia in diabetics, administer insulin, monitor electrolytes, and hydrate.",
        "input": "Nurse Observation: Diabetic patient is lethargic with dry mouth, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Diabetic patient is lethargic with dry mouth\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The diabetic patient has hyperglycemia, with no visible signs, requiring insulin and hydration.\n\nDoctor Prompts\n- Administer insulin as per protocol.\n- Monitor electrolytes for imbalances.\n- Hydrate the patient with fluids."
    },
    {
        "context": "Guideline: For thyroid storm, cool the patient, administer beta-blockers, and give antithyroid medications.",
        "input": "Nurse Observation: Patient has rapid heart rate and fever, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has rapid heart rate and fever\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient may have a thyroid storm, with no visible signs, indicating a hyperthyroid crisis.\n\nDoctor Prompts\n- Cool the patient to reduce temperature.\n- Administer beta-blockers to control heart rate.\n- Give antithyroid medications as prescribed."
    },
    {
        "context": "Guideline: For myxedema coma, administer thyroid hormone, support ventilation, and treat hypotension.",
        "input": "Nurse Observation: Patient is unresponsive with low body temperature, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is unresponsive with low body temperature\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient may have myxedema coma, with no visible signs, indicating a hypothyroid crisis.\n\nDoctor Prompts\n- Administer thyroid hormone immediately.\n- Support ventilation if needed.\n- Treat hypotension with fluids or vasopressors."
    },
    {
        "context": "Guideline: For adrenal crisis, administer hydrocortisone, give IV fluids, and monitor electrolytes.",
        "input": "Nurse Observation: Patient is weak and hypotensive, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is weak and hypotensive\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient may have an adrenal crisis, with no visible signs, requiring steroid replacement.\n\nDoctor Prompts\n- Administer hydrocortisone promptly.\n- Give IV fluids to support blood pressure.\n- Monitor electrolytes for imbalances."
    },
    {
        "context": "Guideline: For anaphylactic shock, administer epinephrine, call emergency services, and monitor airway.",
        "input": "Nurse Observation: Patient has severe swelling and wheezing, Image Analysis: Detected: swelling (Confidence: 0.93)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has severe swelling and wheezing\n- Visual Findings: Detected: swelling (Confidence: 0.93)\n- Summary: The patient is in anaphylactic shock, with visible swelling, indicating a life-threatening allergic reaction.\n\nDoctor Prompts\n- Administer epinephrine immediately.\n- Call emergency services for urgent care.\n- Monitor airway and breathing."
    },
    {
        "context": "Guideline: For septic shock, administer antibiotics, give IV fluids, use vasopressors, and monitor organ function.",
        "input": "Nurse Observation: Patient has fever and low blood pressure, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has fever and low blood pressure\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient may have septic shock, with no visible signs, requiring aggressive treatment.\n\nDoctor Prompts\n- Administer antibiotics promptly.\n- Give IV fluids to support circulation.\n- Use vasopressors and monitor organ function."
    },
    {
        "context": "Guideline: For cardiogenic shock, stabilize with inotropes, consider revascularization, and monitor ECG.",
        "input": "Nurse Observation: Patient has chest pain and weak pulse, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has chest pain and weak pulse\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient may have cardiogenic shock, with no visible signs, indicating a cardiac emergency.\n\nDoctor Prompts\n- Stabilize with inotropes as needed.\n- Consider revascularization if indicated.\n- Monitor ECG for arrhythmias."
    },
    {
        "context": "Guideline: For sudden vision loss, assess for trauma or stroke, keep patient calm, and call emergency services.",
        "input": "Nurse Observation: Patient reports sudden vision loss, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient reports sudden vision loss\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has sudden vision loss, with no visible signs, suggesting a possible stroke or ocular emergency.\n\nDoctor Prompts\n- Assess for trauma or stroke as the cause.\n- Keep the patient calm to reduce stress.\n- Call emergency services immediately."
    },
    {
        "context": "Guideline: For severe headache with vomiting, check for meningitis signs, order a CT scan, and prepare for lumbar puncture if safe.",
        "input": "Nurse Observation: Patient has severe headache and vomiting, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has severe headache and vomiting\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has a severe headache with vomiting, with no visible signs, suggesting possible meningitis.\n\nDoctor Prompts\n- Check for meningitis signs like neck stiffness.\n- Order a CT scan to rule out brain abnormalities.\n- Prepare for lumbar puncture if safe."
    },
    {
        "context": "Guideline: For persistent vomiting, hydrate with small sips, monitor electrolytes, and seek medical help if dehydration occurs.",
        "input": "Nurse Observation: Patient has been vomiting for hours, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has been vomiting for hours\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has persistent vomiting, with no visible signs, indicating a risk of dehydration.\n\nDoctor Prompts\n- Hydrate with small sips of water or electrolyte solution.\n- Monitor electrolytes for imbalances.\n- Seek medical help if dehydration occurs."
    },
    {
        "context": "Guideline: For severe dehydration, administer IV fluids, monitor vital signs, and correct electrolyte imbalances.",
        "input": "Nurse Observation: Patient is lethargic with dry skin, Image Analysis: Detected: dry skin (Confidence: 0.87)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient is lethargic with dry skin\n- Visual Findings: Detected: dry skin (Confidence: 0.87)\n- Summary: The patient has severe dehydration, with visible dry skin, requiring IV fluid resuscitation.\n\nDoctor Prompts\n- Administer IV fluids immediately.\n- Monitor vital signs closely.\n- Correct electrolyte imbalances as needed."
    },
    {
        "context": "Guideline: For kidney stone pain, provide pain relief, order a CT scan, and hydrate the patient.",
        "input": "Nurse Observation: Patient has severe flank pain, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has severe flank pain\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has severe flank pain, likely from a kidney stone, with no visible signs, requiring imaging and pain management.\n\nDoctor Prompts\n- Provide pain relief as needed.\n- Order a CT scan to confirm kidney stones.\n- Hydrate the patient to aid stone passage."
    },
    {
        "context": "Guideline: For urinary retention, insert a catheter, check for infection, and assess for obstruction.",
        "input": "Nurse Observation: Patient cannot urinate and has lower abdominal pain, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient cannot urinate and has lower abdominal pain\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient has urinary retention, with no visible signs, indicating a need for relief and evaluation.\n\nDoctor Prompts\n- Insert a catheter to relieve retention.\n- Check for infection via urine analysis.\n- Assess for obstruction causing the issue."
    },
    {
        "context": "Guideline: For severe nosebleed, pinch nose, lean forward, and seek help if bleeding persists beyond 20 minutes.",
        "input": "Nurse Observation: Patient has a severe nosebleed, Image Analysis: Detected: blood (Confidence: 0.90)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a severe nosebleed\n- Visual Findings: Detected: blood (Confidence: 0.90)\n- Summary: The patient has a severe nosebleed, with visible blood, requiring control and monitoring.\n\nDoctor Prompts\n- Pinch the nose and lean the patient forward.\n- Seek help if bleeding persists beyond 20 minutes.\n- Monitor for excessive blood loss."
    },
    {
        "context": "Guideline: For testicular torsion, prepare for surgery, provide pain relief, and confirm with ultrasound.",
        "input": "Nurse Observation: Patient has sudden testicular pain, Image Analysis: Detected: no visible abnormalities",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has sudden testicular pain\n- Visual Findings: Detected: no visible abnormalities\n- Summary: The patient may have testicular torsion, with no visible signs, requiring urgent surgical evaluation.\n\nDoctor Prompts\n- Prepare for surgery immediately.\n- Provide pain relief as needed.\n- Confirm with ultrasound to guide intervention."
    },
    {
        "context": "Guideline: For severe joint swelling, immobilize, apply ice, and order imaging to rule out septic arthritis.",
        "input": "Nurse Observation: Patient has a swollen and painful knee, Image Analysis: Detected: swelling (Confidence: 0.91)",
        "output": "Patient Report\n- Date: March 22, 2025\n- Nurse Observations: Patient has a swollen and painful knee\n- Visual Findings: Detected: swelling (Confidence: 0.91)\n- Summary: The patient has severe joint swelling, confirmed by imaging, suggesting possible septic arthritis.\n\nDoctor Prompts\n- Immobilize the joint to reduce movement.\n- Apply ice to decrease swelling.\n- Order imaging to rule out septic arthritis."
    },
]


# Combine context for embeddings
guidelines = [entry["context"] for entry in data]
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = sentence_model.encode(guidelines, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Initialize OpenAI client
try:
    client = OpenAI(api_key="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")  # Replace with your API key
except Exception as e:
    print(f"Error initializing OpenAI client: {str(e)}")
    client = None

# Real image analysis function using the SavedModel
def analyze_image(image_path):
    try:
        if image_path is None or image_path == "":
            return "No image provided"

        if model is None or infer is None or not class_names:
            return "Error: SavedModel or labels not loaded"

        # Create the array of the right shape for the model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Convert the input to a TensorFlow tensor and pass it with the correct input name
        input_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
        prediction = infer(sequential_1_input=input_tensor)
        
        # The output key depends on the model's signature
        output_key = list(prediction.keys())[0]
        prediction = prediction[output_key].numpy()

        # Get the predicted class and confidence
        index = np.argmax(prediction[0])
        class_name = class_names[index].strip()
        if " " in class_name:
            class_name = class_name.split(" ", 1)[1]
        confidence_score = prediction[0][index]

        return f"Detected: {class_name} (Confidence: {confidence_score:.2f})"
    except Exception as e:
        return f"Error in image analysis: {str(e)}"

# RAG and report generation with error handling
def generate_report(image_path=None, nurse_observations="No observations recorded"):
    try:
        # Step 1: Real image analysis using the SavedModel
        image_findings = analyze_image(image_path)
        print(f"Image Findings: {image_findings}")

        # Step 2: Use the provided nurse observations
        nurse_obs = nurse_observations if nurse_observations else "No observations recorded"
        print(f"Nurse Observations: {nurse_obs}")

        # Check for mismatch between image findings and nurse observations
        if "Error" not in image_findings and "No image" not in image_findings:
            detected_condition = image_findings.split("Detected: ")[1].split(" (Confidence")[0].lower()
            if "deep cut" in nurse_obs.lower() and "deep cut" not in detected_condition.lower():
                image_findings += " (Note: Image finding may not align with nurse observation of a deep cut)"

        # Step 3: Retrieve guideline with FAISS
        query = f"{nurse_obs} {image_findings}" if "Error" not in image_findings else nurse_obs
        query_embedding = sentence_model.encode([query], convert_to_numpy=True)
        D, I = index.search(query_embedding, k=3)  # Retrieve top 3 guidelines
        retrieved_guidelines = [guidelines[i] for i in I[0]]
        print(f"Retrieved Guidelines: {retrieved_guidelines}")

        # Step 4: Generate report with OpenAI
        if client is None:
            return "Error: OpenAI client not initialized. Please set a valid API key."

        prompt = f"""
        You are a medical assistant in an ER. Using the guidelines, nurse observations, and image findings below, select the most appropriate guideline and generate a comprehensive report for the doctor about the patient's condition. Include a summary of the condition and probable treatment solutions based strictly on the selected guideline. The date must be March 22, 2025, and do not use any other date. Format the response as follows in JSON format:
        Patient Report:
        - Date: March 22, 2025
        - Nurse Observations: [observations]
        - Visual Findings: [findings]
        - Summary: [summary]
        Probable Treatment Solutions:
        - [solution 1]
        - [solution 2]
        - [solution 3]

        Guidelines: {retrieved_guidelines}
        Nurse Observations: "{nurse_obs}"
        Image Findings: "{image_findings}"
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        if not result.startswith("Patient Report:"):
            result = f"Patient Report:\n- Date: March 22, 2025\n- Nurse Observations: {nurse_obs}\n- Visual Findings: {image_findings}\n- Summary: {result.split('Summary:')[-1].split('Probable Treatment Solutions:')[0].strip()}\nProbable Treatment Solutions:\n" + "\n".join([f"- {line.strip()}" for line in result.split("Probable Treatment Solutions:")[-1].split("\n") if line.strip()])
        return result

    except Exception as e:
        return f"Error: {str(e)}"

# Flask endpoint for analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    # Handle image upload
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files['image']
    image_path = os.path.join("uploads", image_file.filename)
    os.makedirs("uploads", exist_ok=True)
    image_file.save(image_path)

    # Get nurse observations (required, since speech-to-text is handled by Golang)
    nurse_observations = request.form.get('nurse_observations', None)
    if not nurse_observations:
        return jsonify({"error": "Nurse observations are required"}), 400

    # Generate report
    report = generate_report(image_path=image_path, nurse_observations=nurse_observations)

    # Clean up the uploaded image
    os.remove(image_path)

    if "Error" in report:
        return jsonify({"error": report}), 500
    return jsonify({"report": report})

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)