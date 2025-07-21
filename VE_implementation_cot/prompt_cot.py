cot_prompt = '''
Instructions:
1. You must always answer using the following chain-of-thought format.
2. Explicitly state the final answer in the form: "The answer is <final_answer>."
3. Ensure the final answer is a single word or entity (not a complete sentence).
4. Refer to the examples below to guide your responses.

Examples:

Q: This British racing driver came in third at the 2014 Bahrain GP2 Series round and was born in what year 
A: First, at the 2014 Bahrain GP2 Series round, DAMS driver Jolyon Palmer came in third. Second, Jolyon Palmer (born 20 January 1991) is a British racing driver. The answer is 1991. 

Q: What band did Antony King work with that formed in 1985 in Manchester? 
A: First, Antony King worked as house engineer for Simply Red. Second, Simply Red formed in 1985 in Manchester. The answer is Simply Red.  

Q: How many inhabitants were in the city close to where Alberta Ferretti’s studios was located? 
A: First, Alberta Ferretti’s studio is near Rimini. Second, Rimini is a city of 146,606 inhabitants. The answer is 146,606. 

Q: TLC: Tables, Ladders & Chairs was a wrestling event featuring which American wrestler and rap- per in the main event? 
A: First, TLC: Tables, Ladders & Chairs was a wrestling event featuring John Cena in the main event. Second, John Cena is an American wrestler and rapper. The answer is John Felix Anthony Cena. 

Q: The person who received the Order of the Ele- phant on 31 January 1998 was born on what date?
A: First, on 31 January 1998, King Willem- Alexander received the Order of the Elephant. Sec- ond, Willem-Alexander was born on 27 April 1967. The answer is 27 April 1967. 

Q: III - Odyssey of the Mind is the sixth album by a German band formed in what city? 
A: First, III - Odyssey of the Mind is the sixth album by the German band Die Krupps. Second, Die Krupps is formed in Düsseldorf. The answer is Düsseldorf. 	
'''

cot_prompt_org ='''
Q: This British racing driver came in third at the 2014 Bahrain GP2 Series round and was born in what year 

A: First, at the 2014 Bahrain GP2 Series round, DAMS driver Jolyon Palmer came in third. Second, Jolyon Palmer (born 20 January 1991) is a British racing driver. The answer is 1991. Q: What band did Antony King work with that formed in 1985 in Manchester? 
A: First, Antony King worked as house engineer for Simply Red. Second, Simply Red formed in 1985 in Manchester. The answer is Simply Red.  

Q: How many inhabitants were in the city close to where Alberta Ferretti’s studios was located? 

A: First, Alberta Ferretti’s studio is near Rimini. Second, Rimini is a city of 146,606 inhabitants. The answer is 146,606. 

Q: TLC: Tables, Ladders & Chairs was a wrestling event featuring which American wrestler and rap- per in the main event? 

A: First, TLC: Tables, Ladders & Chairs was a wrestling event featuring John Cena in the main event. Second, John Cena is an American wrestler and rapper. The answer is John Felix Anthony Cena. 

Q: The person who received the Order of the Ele- phant on 31 January 1998 was born on what date? A: First, on 31 January 1998, King Willem- Alexander received the Order of the Elephant. Sec- ond, Willem-Alexander was born on 27 April 1967. The answer is 27 April 1967. 

Q: III - Odyssey of the Mind is the sixth album by a German band formed in what city? 

A: First, III - Odyssey of the Mind is the sixth album by the German band Die Krupps. Second, Die Krupps is formed in Düsseldorf. The answer is Düsseldorf. 

'''