self_ask_prompt = '''
Instructions:
- For each question, determine if follow-up questions are needed.
- If yes, break it down into simpler questions and gather intermediate answers.
- Ensure the final answer is a single word or entity (not a complete sentence).
- If no follow-up is required, answer directly with one word/entity.

Examples:

Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins? 
A: Are follow up questions needed here: Yes.
Follow up: How old was Theodor Haecker when he died?
Intermediate answer: Theodor Haecker was 65 years old when he died.
Follow up: How old was Harry Vaughan Watkins when he died?
Intermediate answer: Harry Vaughan Watkins was 69 years old when he died.
So the final answer is: Watkins.

Question: Why did the founder of Versus die?
A: Are follow up questions needed here: Yes. 
Follow up: Who founded Versus? 
Intermediate answer: Gianni Versace was the founder of Versus.
Follow up: Why did Gianni Versace die?
Intermediate answer: Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997.
So the final answer is: Shot.

Question: Who is the grandchild of Dambar Shah?
A: Are follow up questions needed here: Yes.
Follow up: Who is the child of Dambar Shah?
Intermediate answer: Dambar Shah (? - 1645) was the king of the Gorkha Kingdom. He was the father of Krishna Shah.
Follow up: Who is the child of Krishna Shah?
Intermediate answer: Krishna Shah (? - 1661) was the king of the Gorkha Kingdom. He was the father of Rudra Shah.
So the final answer is: Rudra Shah.

Question: Are both director of film FAQ: Frequently Asked Questions and director of film The Big Money from the same country?
A: Are follow up questions needed here: Yes.
Follow up: Who directed the film FAQ: Frequently Asked Questions?
Intermediate answer: Carlos Atanes was the director of the film FAQ: Frequently Asked Questions.
Follow up: Who directed the film The Big Money?
Intermediate answer: John Paddy Carstairs directed the movie Big Money.
Follow up: What is the nationality of Carlos Atanes?
Intermediate answer: Carlos Atanes is Spanish.
Follow up: What is the nationality of John Paddy Carstairs? 
Intermediate answer: John Paddy Carstairs is British.
So the final answer is: No.

Question: Which film was released earlier, Navavadhu or The January Man?
A: Are follow up questions needed here: No.
So the final answer is: Navavadhu.

Question: Did Sam Barsky and Linda Vaughn share the same nationality?
A: Are follow up questions needed here: No.
So the final answer is: yes
'''

self_ask_prompt_old = '''
Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins? 
A: Are follow up questions needed here: Yes.
Follow up: How old was Theodor Haecker when he died?
Intermediate answer: Theodor Haecker was 65 years old when he died.
Follow up: How old was Harry Vaughan Watkins when he died?
Intermediate answer: Harry Vaughan Watkins was 69 years old when he died.
So the final answer is: Harry Vaughan Watkins.
Question: Why did the founder of Versus die?
A: Are follow up questions needed here: Yes. 
Follow up: Who founded Versus? 
Intermediate answer: Gianni Versace was the founder of Versus.
Follow up: Why did Gianni Versace die?
Intermediate answer: Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997.
So the final answer is: Shot.
Question: Who is the grandchild of Dambar Shah?
A: Are follow up questions needed here: Yes.
Follow up: Who is the child of Dambar Shah?
Intermediate answer: Dambar Shah (? - 1645) was the king of the Gorkha Kingdom. He was the father of Krishna Shah.
Follow up: Who is the child of Krishna Shah?
Intermediate answer: Krishna Shah (? - 1661) was the king of the Gorkha Kingdom. He was the father of Rudra Shah.
So the final answer is: Rudra Shah.
Question: Are both director of film FAQ: Frequently Asked Questions and director of film The Big Money from the same country?
A: Are follow up questions needed here: Yes.
Follow up: Who directed the film FAQ: Frequently Asked Questions?
Intermediate answer: Carlos Atanes was the director of the film FAQ: Frequently Asked Questions.
Follow up: Who directed the film The Big Money?
Intermediate answer: John Paddy Carstairs directed the movie Big Money.
Follow up: What is the nationality of Carlos Atanes?
Intermediate answer: Carlos Atanes is Spanish.
Follow up: What is the nationality of John Paddy Carstairs? 
Intermediate answer: John Paddy Carstairs is British.
So the final answer is: No.
'''