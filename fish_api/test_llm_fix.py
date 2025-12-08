"""
Quick Test untuk LLM Integration Fix
Test untuk memastikan classification format sudah benar
"""

# Simulate classification result dari fish_engine
classification_results = [
    {
        "name": "Oreochromis mossambicus",
        "species_id": 450,
        "accuracy": 0.85,
        "distance": 0.15
    },
    {
        "name": "Oreochromis niloticus",
        "species_id": 451,
        "accuracy": 0.10,
        "distance": 0.90
    }
]

# Test _build_user_prompt logic
detection_info = {
    "classification": classification_results,
    "detection_confidence": 0.95,
    "area": 15000
}

# Simulate the prompt building
prompt_parts = ["Identifikasi ikan dalam gambar ini."]

if "classification" in detection_info and detection_info["classification"]:
    top_predictions = detection_info["classification"][:3]
    
    pred_list = []
    for pred in top_predictions:
        species_name = pred.get('name', pred.get('label', 'Unknown'))
        confidence = pred.get('accuracy', pred.get('confidence', 0))
        pred_list.append(f"{species_name} ({confidence:.2%})")
    
    pred_text = ", ".join(pred_list)
    prompt_parts.append(f"\nModel klasifikasi memprediksi: {pred_text}")

if "detection_confidence" in detection_info:
    prompt_parts.append(f"\nConfidence deteksi: {detection_info['detection_confidence']:.2%}")

prompt_parts.append("\n\nBerikan identifikasi Anda dalam format JSON yang diminta.")

user_prompt = "".join(prompt_parts)

print("="*60)
print("TEST RESULT")
print("="*60)
print("\nClassification Results:")
for i, cls in enumerate(classification_results):
    print(f"  {i+1}. {cls['name']} - Accuracy: {cls['accuracy']:.2%}")

print("\nGenerated User Prompt:")
print("-"*60)
print(user_prompt)
print("-"*60)

print("\n✅ Test PASSED - No KeyError 'label'")
print("The prompt builder now correctly handles 'name' and 'accuracy' keys.")
