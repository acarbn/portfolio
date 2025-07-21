from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from text_processor import process_text

model_path = "news_summary/saved_model_bart_trainer_epoch6_noclearlabel"

tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

texts = [(
            "Gen X is having a Mounjaro midlife crisis and are paying the consequences. " 
            "A little weight gain during the menopause is normal, but when it happens to a generation "
            "who came of age in the skinny-chic Nineties and beach-body-ready Noughties, the judgement "
            "is real. It is leading to some sad consequences in the world of weight loss jabs, says."
        ),

        (
            "Parts of southern England are set to be battered by torrential rain on Saturday which could"
            "cause “significant” flooding and a danger to life, the Met Office said. An amber warning for"
            " thunderstorms has been issued for between 4am and 11am spanning major towns and cities including "
            "London, Brighton, Portsmouth, Chelmsford, St Albans and Cambridge. Forecasters have warned of "
            "sudden flooding in roads and homes with some more remote communities at risk of being cut off, "
            "while delays to train and bus services are also likely. Power cuts could also occur and buildings "
            "are at risk of damage from floodwater, lightning strikes, hail and strong winds. It is one of "
            "several weather warnings for thunderstorms issued across the country."
        ),

        (   
            "Professional footballers sometimes use a mathematical strategy to help them score a penalty, "
            "or save one – and it all comes down to randomness. As we reach the latter stages of any major "
            "football tournament, penalty shootouts between evenly matched teams seem almost to be an "
            "inevitability. An absorbing spectacle for neutral watchers, agonising for the fans of the teams "
            "involved, and potentially career-defining for the players – the penalty shootout offers a form "
            "of sporting drama almost unrivalled in its acute tension. And so it transpired on Thursday night, "
            "when England and Sweden couldn't be separated over normal and then extra time in their Euro 2025 "
            "quarter final (read more analysis of the game on BBC Sport). Up stepped Allessia Russo for England "
            "to take the first kick putting it just beyond the reach of the diving Swedish keeper Jennifer Falk. "
            "But England would not score again for the next three kicks. Falk dived the right way and saved all "
            "three. When the camera cut to Falk before England's fifth penalty she could be seen consulting her "
            "water bottle on which the names of the England penalty takers and their preferred penalty direction "
            "were listed."
        ),
        (
            "The Paris Opera wants you to 'feel first, understand later'. One of the world's most iconic cultural "
            "institutions, the Paris Opera, invites audiences to experience emotion before intellect. Founded by "
            "Louis XIV in 1669, it has shaped centuries of artistic expression, balancing opera and ballet in equal "
            "measure. But behind the grandeur of the Palais Garnier lies a simple truth, your first connection to "
            "art doesn't need expertise, just openness. Whether you're drawn to theatre, sculpture, or sound, there's "
            "an opera waiting to resonate with you. The experience begins, not just on stage, but the moment you step "
            "inside."
         ),
        (
            "A number of Russian spies have been sanctioned for conducting a 'sustained campaign of malicious cyber activity' "
            "including in the UK, the Foreign Office has said. Three military intelligence units from Russia's GRU espionage "
            "agency and 18 officers have had sanctions placed on them for allegedly 'spreading chaos and disorder on "
            "[Russian President Vladimir] Putin's orders'. UK Foreign Secretary David Lammy linked the activity to the "
            "UK's continued support of Ukraine, and said GRU spies were 'running a campaign to destabilise Europe'. "
            "Separately, the European Union placed its 'strongest sanctions' yet on Russia, which Ukrainian President "
            "Volodymyr Zelensky called 'essential and timely'. The latest EU measures, announced on Friday, included "
            "a ban on transactions related to the Nord Stream natural gas pipeline and lowering a cap on the price at "
            "which Russian oil can be bought. The UK joined the move to lower the price cap, with Chancellor Rachel Reeves "
            "saying Europe was 'turning the screw on the Kremlin's war chest'. They come as European allies hope to ratchet "
            "up the pressure on Russia to bring the three-year-long war in Ukraine to an end. But former Russian President " 
            "Dmitry Medvedev, a close ally of Putin, said his nation's economy would survive the sanctions and that Moscow " 
            "will continue striking Ukraine 'with increasing force'. The EU sanctions are the 18th round of such measures "
            "since Russia's full-scale invasion of Ukraine began in 2022. The aim is to undermine Moscow's ability to finance " 
            "its war on Ukraine - something Western sanctions have so far failed to achieve, as Russia has increased its "
            "oil exports to China and India and operates a so-called shadow fleet of oil tankers around the globe."
        )
    ]

with open("news_summary/saved_model_bart_trainer_epoch6_noclearlabel/summaries_output.txt", "w", encoding="utf-8") as f:
    for idx, text in enumerate(texts):
        clean_text = process_text(text)  # your cleaning function
        input_text = "summarize: " + clean_text

        # Tokenization
        inputs = tokenizer([input_text], max_length=512, truncation=True, return_tensors="pt")

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate summary
        summary_ids = model.generate(
            **inputs,
            max_length=150,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

        # Decode
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Write to file
        f.write(f"\n--- Full Article {idx+1} ---\n")
        f.write(clean_text + "\n")
        f.write(f"\n--- Summary {idx+1} ---\n")
        f.write(summary + "\n\n")
