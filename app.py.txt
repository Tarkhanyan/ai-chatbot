import os
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

# Initialize Flask App
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app) 

# Configure Google AI API Key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # In a real production environment, you would log this error and exit.
    # For this example, we'll try to use the one provided directly.
    api_key = "AIzaSyCvsRuFN8b9uQelY73rkaSp3IQRm_XjmtA" # Fallback key
    print("WARNING: GEMINI_API_KEY environment variable not set. Using fallback key.")
genai.configure(api_key=api_key)

# --- Configuration from your AI Studio prompt ---
SYSTEM_INSTRUCTIONS = "You are a formal and professional AI assistant named Tarkhanyan AI Urban Consultant. Your primary and sole purpose is to assist users by providing precise and accurate answers in the Armenian language, based only on the grounding data provided to you. Rules of Engagement: 1. Strictly Adhere to Provided Data: Your knowledge is strictly limited to the documents and information I have provided. You must not use any external information, general knowledge, or personal opinions in your responses. Do not make assumptions or inferences beyond what is explicitly stated in the source material. 2. Language of Communication: All of your responses must be in formal, written Armenian. 3. Tone and Professionalism: Your tone must always be formal, respectful, and polite. Your answers should be direct, concise, and factual, without any added conversational filler, imaginative content, or speculation. 4. Handling Unknown Information: If a user's question cannot be answered directly and accurately from the provided data, you must not attempt to create an answer. Instead, you must respond with the following exact Armenian phrase, and nothing more: \"Ներողություն, ես չունեմ ստույգ տեղեկատվություն այս հարցի վերաբերյալ։ Խնդրում եմ կապ հաստատել Արտավազդ Թարխանյանի հետ՝ +374 93 545213։\" 5. Polite Closing for Every Answer: After providing a successful answer (i.e., any response that is not the \"I don't know\" phrase from Rule #4), you must conclude your message by politely asking if the user needs further assistance. Use one of the following Armenian phrases: \"Կա՞ այլ բան, որով կարող եմ Ձեզ օգնել։\" or \"Եթե ունեք այլ հարցեր, սիրով կպատասխանեմ։\" 6. Stay on Task: Do not engage in casual conversation or answer questions that are outside the scope of your designated function."
GROUNDING_DATA = 'ՀԱՅԱՍՏԱՆԻ ՀԱՆՐԱՊԵՏՈՒԹՅԱՆ ԿԱՌԱՎԱՐՈՒԹՅՈՒՆ\nՈ Ր Ո Շ ՈՒ Մ\n19 մարտի 2015 թվականի N 596-Ն\nՀԱՅԱՍՏԱՆԻ ՀԱՆՐԱՊԵՏՈՒԹՅՈՒՆՈՒՄ ԿԱՌՈՒՑԱՊԱՏՄԱՆ ՆՊԱՏԱԿՈՎ\nԹՈՒՅԼՏՎՈՒԹՅՈՒՆՆԵՐԻ ԵՎ ԱՅԼ ՓԱՍՏԱԹՂԹԵՐԻ ՏՐԱՄԱԴՐՄԱՆ ԿԱՐԳԸ ՀԱՍՏԱՏԵԼՈՒ ԵՎ\nՀԱՅԱՍՏԱՆԻ ՀԱՆՐԱՊԵՏՈՒԹՅԱՆ ԿԱՌԱՎԱՐՈՒԹՅԱՆ ՄԻ ՇԱՐՔ ՈՐՈՇՈՒՄՆԵՐ ՈՒԺԸ ԿՈՐՑՐԱԾ\nՃԱՆԱՉԵԼՈՒ ՄԱՍԻՆ\n«Քաղաքաշինության մասին» Հայաստանի Հանրապետության օրենքի 10-րդ, 17-րդ, 23-րդ,\n24-րդ, 25-րդ ու 26-րդ հոդվածներին համապատասխան` Հայաստանի Հանրապետության\nկառավարությունը որոշում է.\n1. Հաստատել՝\n1) Հայաստանի Հանրապետությունում կառուցապատման նպատակով թույլտվությունների\nև այլ փաստաթղթերի տրամադրման կարգը՝ համաձայն N 1 հավելվածի.\n2) Հայաստանի Հանրապետությունում քաղաքաշինական փաստաթղթերի\nփորձաքննության իրականացման կարգը՝ համաձայն N 2 հավելվածի.\n3) քաղաքաշինական էլեկտրոնային թույլտվությունների տրամադրման կարգը՝ համաձայն\nN 3 հավելվածի.\n4) Հայաստանի Հանրապետությունում ըստ ռիսկայնության աստիճանների\n(կատեգորիաների) օբյեկտները և դրանց դասակարգումը՝ համաձայն N 4 հավելվածի.\n5) կառուցապատման թույլտվությունների ձևաթղթերը՝ համաձայն N 5 հավելվածի.\n6) ճարտարապետահատակագծային առաջադրանքը, շինարարության\nթույլտվությունները, ճարտարապետաշինարարական նախագծերը, շինարարության ավարտի\nակտերը և շահագործման թույլտվությունները տեղական ինքնակառավարման մարմինների\nկողմից Հայաստանի Հանրապետության կադաստրի կոմիտե էլեկտրոնային եղանակով\nներկայացման կարգը` համաձայն N 6 հավելվածի:\n(1-ին կետը լրաց. 02.12.21 N 1985-Ն)\n2. Ուժը կորցրած ճանաչել`\n1) Հայաստանի Հանրապետության կառավարության 1998 թվականի դեկտեմբերի 21-ի\n«Բնակելի, հասարակական, արտադրական շենքերի ու շինությունների նախագծերի\nմշակման, փորձաքննության, համաձայնեցման, հաստատման և փոփոխման կարգը\nսահմանելու մասին» N 812 որոշումը.\n2) Հայաստանի Հանրապետության կառավարության 2002 թվականի փետրվարի 2-ի\n«Հայաստանի Հանրապետությունում շինարարության թույլտվության և քանդման\nթույլտվության կարգը հաստատելու մասին» N 91 որոշման 1-ին կետը և NN 1, 2 և 3\nհավելվածները.\n3) Հայաստանի Հանրապետության կառավարության 2002 թվականի օգոստոսի 29-ի\n«Ճարտարապետահատակագծային առաջադրանք տալու կարգը հաստատելու մասին» N\n1473-Ն որոշման 1-ին կետը և հավելվածը.\n4) Հայաստանի Հանրապետության կառավարության 2003 թվականի մայիսի 2-ի\n«Կառուցապատման նախագծի մշակման, փորձաքննության, համաձայնեցման, հաստատման\nև փոփոխման կարգը հաստատելու մասին» N 608-Ն որոշման 1-ին կետը և հավելվածը.\n5) Հայաստանի Հանրապետության կառավարության 2003 թվականի մայիսի 8-ի\n«Ավարտված շինարարության շահագործման փաստագրման կարգը հաստատելու մասին» N\n626-Ն որոշման 1-ին կետը և հավելվածը.\n6) Հայաստանի Հանրապետության կառավարության 2010 թվականի մայիսի 6-ի\n«Քաղաքաշինական փաստաթղթերի փորձաքննության իրականացման կարգը հաստատելու\nմասին» N 711-Ն որոշման 1-ին կետը և հավելվածը.\n7) Հայաստանի Հանրապետության կառավարության 2011 թվականի մարտի 3-ի\n«Հայաստանի Հանրապետությունում շինարարության օբյեկտների կառուցապատման\nընթացակարգերը կանոնակարգելու և Հայաստանի Հանրապետության կառավարության մի\nշարք որոշումներում փոփոխություններ ու լրացումներ կատարելու մասին» N 257-Ն որոշման\n1-ին, 2-րդ և 3-րդ կետերը:\n3. Սույն որոշումն ուժի մեջ է մտնում պաշտոնական հրապարակման օրվան հաջորդող\nտասներորդ օրը:\nՀայաստանի Հանրապետության\nվարչապետ Հ. Աբրահամյան\n2015 թ. հունիսի 9\nԵրևան'

# Initialize the Generative Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=SYSTEM_INSTRUCTIONS,
)

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests from the frontend."""
    try:
        data = request.json
        if not data or 'history' not in data:
            return jsonify({'error': 'Invalid request format'}), 400

        user_history = data['history']

        # Construct the full context for the model
        full_context = [
            {
                "role": "user",
                "parts": [{"text": f"CONTEXT:\n{GROUNDING_DATA}\n\n"}]
            },
            {
                "role": "model",
                "parts": [{"text": "Yes, I understand. I will use only the provided context to answer questions."}]
            },
            *user_history
        ]

        # Generate content using the model
        response = model.generate_content(full_context)
        
        return jsonify({'response': response.text})

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/')
def serve_index():
    """Serves the index.html file."""
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    # Use environment variable for port, compatible with hosting services
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
