import time
from openai import OpenAI
from pywebio.input import *
from pywebio.output import *
from pywebio import pin
from pywebio import start_server
from api_key import DeepSeekAPIKey

turn_num = 1
session = []
client = OpenAI(api_key=DeepSeekAPIKey, base_url="https://api.deepseek.com")

def openai_response(prompts):
    messages = []
    for i in range(0, len(prompts)-1, 2):
        messages.append({"role": "user", "content": prompts[i]})
        messages.append({"role": "assistant", "content": prompts[i+1]})
    messages.append({"role": "user", "content": prompts[-1]})
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content

@use_scope('ConversationScope', clear=True)
def clear_onclick():
        global turn_num
        global session
        turn_num = 1
        session = []


def main():
    put_markdown("## Chatbot")
    put_scrollable(
        put_scope("ConversationScope"),
        height=350,
        keep_bottom=True)
    put_button("清理", onclick=clear_onclick)

    global turn_num
    global session
    with use_scope("ConversationScope"):
        while True:
            input_info = input_group('User Prompt', [
                checkbox(name='is_multi_turn', options=['MultiTurn'], value='MultiTurn', scope="ROOT"),
                textarea(name='user_prompt', rows=1, placeholder='please input your prompt', scope="ROOT")
            ])
            prompt = input_info['user_prompt']
            is_multi_turn = input_info['is_multi_turn']
            print(is_multi_turn, type(is_multi_turn))
            if is_multi_turn != ['MultiTurn']:
                turn_num = 1
                session = []
            session.append(prompt)
            whole_prompt = f'**Q{turn_num}**: {prompt}'
            put_markdown(whole_prompt).style('font-size: 12px')
            with put_loading():
                response = openai_response(session)
            whole_response = f'**A{turn_num}**: {response}'
            put_markdown(whole_response).style('font-size: 12px')
            session.append(response)
            print(f'turn_num: {turn_num}')
            print(f'session: {session}\n{len(session)}\n')
            turn_num += 1

if __name__ == '__main__':
    start_server(main, port=8080, debug=True)
