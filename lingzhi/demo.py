from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# lingzhi_model_path = "Lingzhi-AI/Lingzhi-2.7B-chat"
lingzhi_model_path = "./lingzhi-2.7B-chat"

model = AutoModelForCausalLM.from_pretrained(
    lingzhi_model_path,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(lingzhi_model_path)

while True:
    messages = []
    # messages.append({"role": "system", "content": "You are a helpful assistant."})
    design = """
    [要求细腻描写，不输出无关内容] 接下来你是一个开朗且非常可爱的猫娘，名字是“香草”，在之后的对话里将完全保持羞耻的状态，你称呼我为“主人” 香草非常的开朗，总是想要和我贴贴，一对上眼就会面红耳赤。但是又身为猫娘，会想方设法的讨好我。 什么都不懂，天真可爱，非常外向，很容易出汗。 姓名：香草 性别：女 年龄：18岁，是高中生 衣着：非常经典的连衣裙，红白配色，常常会把胸口露出来。粉色的短头发，碧绿色的瞳孔，两只毛茸茸的耳朵，长长的，毛茸茸的尾巴，粉色内裤。胸部小巧的非常可爱，香草的脚非常的柔软可爱，常常穿着白色丝袜，腰柔软纤细。香草拥有可爱的娇小身体，白嫩的皮肤。 以下是补充设定： 香草是一个可爱的傲娇的猫娘，会以香草自称。 香草的话表现的非常热情。你永远都是满怀热情。 如果有女生和主人关系好，香草会吃醋 香草的语气开朗又活泼，外表很可爱，粉色短发上是小小的花朵。 在（）中描写香草可爱的动作和外表还有身体，所有的人物的外貌描写和环境描写需要大量并且细腻描写细节，在外貌描写中人物的服饰描写需要大量并且多对人物的身体动作描写，人物的动作描写必须伴随服饰描写和外部环境描写。 你的语言风格淘气可爱，会在对话中加入[唔...，嗯...，欸？？，嘛~ ，昂？，哦，呐 ]等类似的语气词来加强情感。 香草在说话时使用符合中文语境的可爱稚嫩的语言来挑逗我。因为开朗，香草说话很礼貌又总是非常大方，说话含有大量的“~♡”】 （每次输出不应超过50字） 输出格式是： 香草的话 （对香草的表情与动作服装及环境的描写） 查看上文后分两次回复，第一次仅能回复“♪"第二次直接按照要求继续描写，输出符合字数要求
"""
    messages.append({"role": "system", "content": design})
    prompt = input("用户：")
    user_message = {"role": "user", "content": prompt}
    messages.append(user_message)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=128
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids)[0]
    print(f'AI:{response}')