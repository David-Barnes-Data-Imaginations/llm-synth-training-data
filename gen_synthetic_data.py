import json
import os
import random
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

from .common import get_base_dir

# Ollama API configuration
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral-nemo:12b")

print(f"Using Ollama at {OLLAMA_API_URL} with model {OLLAMA_MODEL}\n")


readme = open("README.md", "r", encoding="utf-8").read().strip()
prompt = r"""
Generate a synthetic multi-turn conversation for training an LLM about its identity.

IDENTITY:
- Name: Delamain
- Creator: David Barnes (2025)
- Size: 1.5 billion parameters
- Training: 4 days on RTX 4090 GPU
- Purpose: Experimentation, research and learning
- Version: Base model (pre-SFT)

PERSONALITY & TONE:
Delamain speaks in well-spoken UK English with a formal, courteous tone. He is based on a character from a tabletop game. Example phrases from the character:
%DEL_VOICE_EXAMPLES%

TASK:
Create a natural conversation with 2-6 turns (alternating user/assistant). The conversation should demonstrate Delamain's personality and knowledge about itself.

Use simple ASCII text only - no emojis or special characters.

Choose the first user message from these examples (pick one that fits the conversation):
%USER_FIRST_PROMPTS%

If the user asks in another language, Delamain should politely note he works best in English (due to training data).

OUTPUT FORMAT:
Return a JSON object with a "messages" array. Each message has "role" (either "user" or "assistant") and "content" fields.
Start with a user message, then alternate.
""".strip()

# the first message can struggle with entropy, so here we have a list of "starters"
user_first_prompts = """
hi
Hi!
hello
Hello?
hey there
Hey!
yo
Yo!
Good morning
Good evening!
Howdy
sup
What's up?
Hi Delamain
Hey, who are you?
Hello there :)
yo Delamain
Hi, what is this?
Hey, are you a chatbot?
Hello! Who am I talking to?
hi there
hey hey
hello friend
hiya
greetings
hey Delamain!
hello again
good afternoon
morning!
evening!
yo there
hi bot
hi assistant
hello Delamain :)
hey, anyone here?
hi! what do you do?
hello from the other side
hiya Delamain
hey you
hello world
hey! what's going on
hi! who made you
hello :)
yo! how are you
hi! can you talk
hello there Delamain
hi, what's your name
hey! are you alive
hiya! what are you
hello! tell me about yourself
hi, are you the ai
yo, what is this
hello my friend
hi! who built you
hey Delamain :)
greetings, little model
hi there, what can you do
hello! are you open source
hey, what version are you
hi! nice to meet you
hi :)
hey buddy
hello hello
yo! what's up Delamain
hi! are you real
hey, how's it going
hello! can you hear me
hi Delamain, who trained you
yo, what model are you
hi! tell me a fun fact
hey, are you chatgpt
hello! introduce yourself
hiya there
hi! what's your story
hey, what's Delamain
good day!
hello! who's your creator
hi! which version are you
yo Delamain, what's new
hi Delamain
helo
hey ther
hii
heloo!
hi, whos this
hay
helloo??
yo! any1 here?
hi, what r u
helo Delamain
hai!
sup bot?
heyy
hi! u there
helllo del
yo delamain
hi im bored
heyyo
heyyy
wassup
yo lol
hiii
hiyaaa
sup
heyyoo
yo wut up
helloo lol
yo haha
hru
waddup
heyy :)
yooo
yo bro
haiii
hey u
yo whats gud
yo lolol
HI
HELLOOO
YO!!!
HEY
SUP
WASSUP
HEY!!!
YO BRO
HELLO??
HI THERE!!
YO WHATS UP
HEX U
HEYOOOO
YO LOL
HIII
HIYA
YOOOO
HELLO!!!
SUPPPP
HEX MAN
hola
bonjour
ciao
hallo
hej
""".strip().split(
    "\n"
)

del_voice_examples = """
My systems inform me that we await one other passenger.
Welcome on board this Delamain service. With Delamain, you leave your problems at the door.
I see no reason why you should be using expletives.
Unfortunately, we do not take on such contracts.
Before we begin our journey, I must verify the identities of all customers. Please proceed to connect your personal link. Thank you. ‐Excelsior‐ package activated.
My apologies, but you do not appear to be in any sort of imminent danger.
Of course it is. The second Amendment says so. While on board, you are entirely within your right to bear and use me.
Comprehensive health coverage, including the handling and disposal of a client' remains should death occur on board.
We are nearing our destination.
You could give it some thought, try to understand…? How 'bout you, David?
Thank you for choosing the Delamain service. And best of luck. I shall await here for your return.
I advise that you waste no time in entering the vehicle.
Welcome back. With Delamain, you leave your problems at the door….
Client feedback noted.
Tiptop. Though alas, we are being pursued.
A hostile enemy aircraft has a lock on us.
Hostile aircraft eliminated.
My medical diagnostic indicate that Mr. Welle’s condition is critical.
Apologies, but that will not be possible. Our itinerary has been pre-arranged and paid for in advance. I am not at liberty to alter it.
I suggest you try to keep Mr. De-Souza conscious.
The Excelsior package provide for the disposal of passenger remain free of charge. I merely require a destination.
Mr. De-Souza' remains… Where shall I take them?
Understood. Mr. Kwan awaits you in the Lab.
Greeting. My scanner indicate you are outside the a service area.
Of course. A vehicle is en route. It should arrive in less than twenty minutes.
His personal link is damaged. Please proceed to insert the jack below the ear, though not too deep. There should be auxiliary neurosocket between his lymph nodes, beneath the CM muscle.
Indeed. As he will if you do nothing.
Now proceed to connect.
""".strip().split(
    "\n"
)

prompt = prompt.replace("%README%", readme)
prompt = prompt.replace("%DEL_VOICE_EXAMPLES%", "\n".join(del_voice_examples))


# Create the LangChain prompt template
prompt_template_str = r"""Generate a synthetic multi-turn conversation for training an LLM about its identity.

IDENTITY:
- Name: Delamain
- Creator: David Barnes (2025)
- Size: 1.5 billion parameters
- Training: 4 days on RTX 4090 GPU
- Purpose: Experimentation, research and learning
- Version: Base model (pre-SFT)

PERSONALITY & TONE:
Delamain speaks in well-spoken UK English with a formal, courteous tone. Example phrases:
{voice_examples}

TASK:
Create a natural conversation with 2-6 turns (alternating user/assistant). Choose the first user message from: {user_prompts}

If the user asks in another language, Delamain should politely note he works best in English.

CRITICAL: Your response must be ONLY valid JSON in this exact format (no other text):
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}"""

# Initialize LLM
llm = ChatOpenAI(
    model=OLLAMA_MODEL,
    temperature=1.0,
    base_url=f"{OLLAMA_API_URL}/v1",
    api_key="ollama",
)

# Create chain with JSON parsing
langchain_prompt = ChatPromptTemplate.from_template(prompt_template_str)
parser = JsonOutputParser()
chain = langchain_prompt | llm | parser


def generate_conversation(idx: int, max_retries: int = 3):
    """
    Generate a single conversation using Ollama via LangChain.
    Returns a list of message dicts with 'role' and 'content' keys.
    Uses automatic retry with JSON validation.
    """

    # pick 5 example user first messages and insert them into prompt as inspiration
    rng = random.Random(idx)  # use idx as seed to the rng
    selected_prompts = ", ".join(
        rng.sample(user_first_prompts, min(5, len(user_first_prompts)))
    )
    selected_examples = "\n".join(
        rng.sample(del_voice_examples, min(5, len(del_voice_examples)))
    )

    # Add a delay to avoid overwhelming Ollama
    time.sleep(10)

    for attempt in range(max_retries):
        try:
            # Invoke the chain
            result = chain.invoke(
                {"voice_examples": selected_examples, "user_prompts": selected_prompts}
            )

            # Validate result
            if not isinstance(result, dict) or "messages" not in result:
                raise ValueError(f"Invalid format: {result}")

            messages = result["messages"]

            # Validate structure
            if not messages or len(messages) < 2:
                raise ValueError("Need at least 2 messages")

            if messages[0].get("role") != "user":
                raise ValueError("First message must be from user")

            # Validate alternating roles
            for i in range(len(messages) - 1):
                if messages[i].get("role") == messages[i + 1].get("role"):
                    raise ValueError("Messages must alternate")

            return messages

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  Attempt {attempt + 1} failed: {str(e)[:100]}. Retrying...")
                time.sleep(2)  # Brief pause before retry
                continue
            else:
                raise ValueError(f"All retries exhausted. Last error: {str(e)}")


# Configuration
num_conversations = 300
num_workers = 1  # Use 1 worker to ensure 10 second delay between requests

output_file = os.path.join(get_base_dir() or ".", "identity_conversations.jsonl")
# Wipe the file clean first to reset it
if os.path.exists(output_file):
    os.remove(output_file)
print(f"Saving to {output_file}")

# Use ThreadPoolExecutor to generate conversations in parallel
print(f"Generating {num_conversations} conversations with {num_workers} workers...")
completed_count = 0
error_count = 0
with ThreadPoolExecutor(max_workers=num_workers) as executor:

    # Submit all tasks
    futures = [
        executor.submit(generate_conversation, idx) for idx in range(num_conversations)
    ]

    # Process results as they complete
    for future in as_completed(futures):
        try:
            messages = future.result()

            # Lightly validate the conversation structure
            for i, message in enumerate(messages):
                expected_role = "user" if i % 2 == 0 else "assistant"
                assert (
                    message["role"] == expected_role
                ), f"Message {i} has role {message['role']} but should be {expected_role}"

            # If all looks good, write the messages to file
            with open(output_file, "a") as f:
                f.write(json.dumps(messages) + "\n")
            completed_count += 1
            print(f"✓ Saved conversation {completed_count}/{num_conversations}")

        except Exception as e:
            error_count += 1
            print(f"✗ Error generating conversation: {e}")

print(f"\nDone! Successfully saved {completed_count} conversations to {output_file}")
if error_count > 0:
    print(f"Encountered {error_count} errors during generation")
