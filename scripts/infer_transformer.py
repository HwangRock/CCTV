import torch
import yaml
import sys
import os
import datasets
from transformers import BertTokenizer, EncoderDecoderModel

def main():
    print('RNN for machine translation evaluation')

    if len(sys.argv) >= 2:
        params_filename = sys.argv[1]
        print(sys.argv)
    else:
        params_filename = '../config/sum_transformer.yaml'

    with open(params_filename, 'r', encoding="UTF8") as f:
        params = yaml.safe_load(f)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    timestamp = "1696999441"
    checkpoint = 'checkpoint-26500'
    out_dir = os.path.abspath((os.path.join(os.path.curdir, "runs", timestamp, checkpoint)))

    tokenizer = BertTokenizer.from_pretrained(params['encoder_model'])
    model = EncoderDecoderModel.from_pretrained(out_dir)
    model.to(device)

    article = "(CNN) Gaming has achieved an unprecedented milestone by being selected as an official medal sport for the Hangzhou Asian Games in the form of esports, but participating in the competition holds significant importance – and possibly life-changing implications – for some players. " \
              "For South Korean male participants, winning a gold medal in the Asian Games or Olympic Games not only brings honor but also grants them a military exemption. Military service is compulsory for men in South Korea, with almost all able-bodied persons required to serve in the army for 18 months by the age of 28. " \
              "However, South Korean law allows men who are deemed to excel in sports, popular culture, art or higher education to defer their service until the age of 30. The law has impacted the careers of some of the country’s biggest names, including global superstar music group BTS. " \
              "Three BTS members are currently serving in the military, with band member Suga having begun his service on September 22. However, the mandatory duty can be waived for some athletes, in particular those who win an Olympic medal or a gold medal in the Asian Games. " \
              "Son Heung-min, captain of Premier League soccer team Tottenham Hotspur, received an exemption from his mandatory military conscription in 2018 after winning gold at the Asian Games with South Korea. " \
              "For esports players, achieving success on the international stage could lead to the same outcome and the possibly unusual situation of Korean men earning a military exemption for playing a video game. " \
              "While exempt athletes are still required to undertake a three- or four-week period of training, this recognition is a significant step forward for the gaming industry in South Korea."

    # article = "(CNN)  -- The National Football League has indefinitely suspended Atlanta Falcons quarterback Michael Vick without pay, officials with the league said Friday. NFL star Michael Vick is set to appear in court Monday. A judge will have the final say on a plea deal. Earlier, Vick admitted to participating in a dogfighting ring as part of a plea agreement with federal prosecutors in Virginia. \"Your admitted conduct was not only illegal, but also cruel and reprehensible. Your team, the NFL, and NFL fans have all been hurt by your actions,\" NFL Commissioner Roger Goodell said in a letter to Vick. Goodell said he would review the status of the suspension after the legal proceedings are over. In papers filed Friday with a federal court in Virginia, Vick also admitted that he and two co-conspirators killed dogs that did not fight well. Falcons owner Arthur Blank said Vick's admissions describe actions that are \"incomprehensible and unacceptable.\" The suspension makes \"a strong statement that conduct which tarnishes the good reputation of the NFL will not be tolerated,\" he said in a statement.  Watch what led to Vick's suspension \u00bb . Goodell said the Falcons could \"assert any claims or remedies\" to recover $22 million of Vick's signing bonus from the 10-year, $130 million contract he signed in 2004, according to The Associated Press. Vick said he would plead guilty to one count of \"Conspiracy to Travel in Interstate Commerce in Aid of Unlawful Activities and to Sponsor a Dog in an Animal Fighting Venture\" in a plea agreement filed at U.S. District Court in Richmond, Virginia. The charge is punishable by up to five years in prison, a $250,000 fine, \"full restitution, a special assessment and 3 years of supervised release,\" the plea deal said. Federal prosecutors agreed to ask for the low end of the sentencing guidelines. \"The defendant will plead guilty because the defendant is in fact guilty of the charged offense,\" the plea agreement said. In an additional summary of facts, signed by Vick and filed with the agreement, Vick admitted buying pit bulls and the property used for training and fighting the dogs, but the statement said he did not bet on the fights or receive any of the money won. \"Most of the 'Bad Newz Kennels' operations and gambling monies were provided by Vick,\" the official summary of facts said. Gambling wins were generally split among co-conspirators Tony Taylor, Quanis Phillips and sometimes Purnell Peace, it continued. \"Vick did not gamble by placing side bets on any of the fights. Vick did not receive any of the proceeds from the purses that were won by 'Bad Newz Kennels.' \" Vick also agreed that \"collective efforts\" by him and two others caused the deaths of at least six dogs. Around April, Vick, Peace and Phillips tested some dogs in fighting sessions at Vick's property in Virginia, the statement said. \"Peace, Phillips and Vick agreed to the killing of approximately 6-8 dogs that did not perform well in 'testing' sessions at 1915 Moonlight Road and all of those dogs were killed by various methods, including hanging and drowning. \"Vick agrees and stipulates that these dogs all died as a result of the collective efforts of Peace, Phillips and Vick,\" the summary said. Peace, 35, of Virginia Beach, Virginia; Phillips, 28, of Atlanta, Georgia; and Taylor, 34, of Hampton, Virginia, already have accepted agreements to plead guilty in exchange for reduced sentences. Vick, 27, is scheduled to appear Monday in court, where he is expected to plead guilty before a judge.  See a timeline of the case against Vick \u00bb . The judge in the case will have the final say over the plea agreement. The federal case against Vick focused on the interstate conspiracy, but Vick's admission that he was involved in the killing of dogs could lead to local charges, according to CNN legal analyst Jeffrey Toobin. \"It sometimes happens -- not often -- that the state will follow a federal prosecution by charging its own crimes for exactly the same behavior,\" Toobin said Friday. \"The risk for Vick is, if he makes admissions in his federal guilty plea, the state of Virginia could say, 'Hey, look, you admitted violating Virginia state law as well. We're going to introduce that against you and charge you in our court.' \" In the plea deal, Vick agreed to cooperate with investigators and provide all information he may have on any criminal activity and to testify if necessary. Vick also agreed to turn over any documents he has and to submit to polygraph tests. Vick agreed to \"make restitution for the full amount of the costs associated\" with the dogs that are being held by the government. \"Such costs may include, but are not limited to, all costs associated with the care of the dogs involved in that case, including if necessary, the long-term care and\/or the humane euthanasia of some or all of those animals.\" Prosecutors, with the support of animal rights activists, have asked for permission to euthanize the dogs. But the dogs could serve as important evidence in the cases against Vick and his admitted co-conspirators. Judge Henry E. Hudson issued an order Thursday telling the U.S. Marshals Service to \"arrest and seize the defendant property, and use discretion and whatever means appropriate to protect and maintain said defendant property.\" Both the judge's order and Vick's filing refer to \"approximately\" 53 pit bull dogs. After Vick's indictment last month, Goodell ordered the quarterback not to report to the Falcons training camp, and the league is reviewing the case. Blank told the NFL Network on Monday he could not speculate on Vick's future as a Falcon, at least not until he had seen \"a statement of facts\" in the case.  E-mail to a friend . CNN's Mike Phelan contributed to this report."

    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(article, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print(output_str)


if __name__ == "__main__":
    main()