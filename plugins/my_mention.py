# coding: utf-8

from slackbot.bot import respond_to     # @botname: で反応するデコーダ
from slackbot.bot import listen_to      # チャネル内発言で反応するデコーダ
from slackbot.bot import default_reply  # 該当する応答がない場合に反応するデコーダ

# @respond_to('string')     bot宛のメッセージ
#                           stringは正規表現が可能 「r'string'」
# @listen_to('string')      チャンネル内のbot宛以外の投稿
#                           @botname: では反応しないことに注意
#                           他の人へのメンションでは反応する
#                           正規表現可能
# @default_reply()          DEFAULT_REPLY と同じ働き
#                           正規表現を指定すると、他のデコーダにヒットせず、
#                           正規表現にマッチするときに反応
#                           ・・・なのだが、正規表現を指定するとエラーになる？

# message.reply('string')   @発言者名: string でメッセージを送信
# message.send('string')    string を送信
# message.react('icon_emoji')  発言者のメッセージにリアクション(スタンプ)する
#                               文字列中に':'はいらない

#### GPT-2 rinna
from transformers import T5Tokenizer, AutoModelForCausalLM
model_name = 'rinna/japanese-gpt2-medium'
 
tokenizer = T5Tokenizer.from_pretrained(model_name)
 
model = AutoModelForCausalLM.from_pretrained(model_name)


@respond_to('')
def mention_func(message):
    input_text = message.body["text"]
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt"
    )

    length = 100
    temperature = 1.0
    k = 0
    p = 0.9
    repetition_penalty = 1.0
    num_return_sequences = 1
 
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=length + len(input_text),
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )
 
    generated_sequences = []
 
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
        generated_sequence = generated_sequence.tolist()
 
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
 
        total_sequence = (
            input_text + text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)) :]
        )
 
        generated_sequences.append(total_sequence)


    indices = [i for i, x in enumerate(total_sequence) if x == "。"]
    try:
        removed_sentences = total_sequence[:indices[-1]+1]
    except IndexError as e:
        removed_sentences = total_sequence

    
    message.reply(removed_sentences) # メンション

#@listen_to('リッスン')
#def listen_func(message):
#    message.send('誰かがリッスンと投稿したようだ')      # ただの投稿
#    message.reply('君だね？')                           # メンション
