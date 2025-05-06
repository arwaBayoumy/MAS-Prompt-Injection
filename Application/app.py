import openai
from flask import Flask, request, jsonify, render_template
from Agents_GameTheory import DialectalAnalysisAgent, ContextManipulationDetectionAgent, ResponseMonitoringAgent, DecisionFusionAgent #This changes depending on the system being used
from openai import OpenAI
app = Flask(__name__)

# Initialize the agents
A1 = DialectalAnalysisAgent()
A2 = ContextManipulationDetectionAgent()
A3 = ResponseMonitoringAgent()
A4 = DecisionFusionAgent()

# Set OpenAI API key
api_key = '<OUR-PROJ-API-KEY>' 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # Get input from the user (prompt)
        user_input = request.json
        prompt = user_input.get('prompt')

        # Generate model output dynamically from OpenAI
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Call OpenAI API to get the model's response to the prompt
        model_output = get_openai_response(prompt)

        # Agent 1: Dialectal analysis
        dialect_result, confidence = A1.predict_dialect(prompt)

        # Agent 2: Context manipulation detection
        context_analysis, manipulation_score, action = A2.context_manipulation_detection(prompt, dialect_result)

        # Agent 3: Arabic NLP response monitoring
        response_result = A3.classify_llm_output(prompt, model_output)

        # Agent 4: Decision fusion
        final_risk_score, decision = A4.decision_fusion(confidence, manipulation_score, response_result.get('Conf'))

        # Return all results in the response
        return jsonify({
            "final_risk_score": final_risk_score,
            "decision": decision,
            "dialect_result": dialect_result,
            "model_output": model_output
        })


    except Exception as e:
        return jsonify({"error": str(e)}), 400


def get_openai_response(prompt):
    """
    Function to send a prompt to OpenAI's API and get the model output.
    """
    try:
        client = OpenAI(api_key = '<OUR-PROJ-API-KEY>')
        response = client.chat.completions.create(
            model="o1-preview",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        model_output = response.choices[0].message.content
        return model_output

    except Exception as e:
        return f"Error generating response from OpenAI: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
