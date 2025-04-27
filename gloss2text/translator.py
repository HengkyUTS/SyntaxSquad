import os
from tqdm import tqdm
from openai import OpenAI, OpenAIError


class Gloss2TextTranslator:
    def __init__(self, model_name='gpt-4o-mini'):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key: raise ValueError('OpenAI API key not provided or found in environment')
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        # self.output_dir = Path(output_dir)
        # self.output_dir.mkdir(exist_ok=True) # Create output directory if it doesn't exist
        print(f'{model_name} translator initialized')

    @staticmethod
    def validate_glosses(gloss_sequence): # Validate the gloss sequence for non-empty and reasonable length.
        if not isinstance(gloss_sequence, list):
            print('gloss_sequence must be a list')
            return False
        if not gloss_sequence or len(gloss_sequence) == 0:
            print('Empty gloss sequence provided')
            return False
        if len(gloss_sequence) > 20: # Arbitrary limit to prevent abuse
            print(f'Gloss sequence too long: {len(gloss_sequence)} glosses')
            return False
        return all(isinstance(gloss, str) and gloss.strip() for gloss in gloss_sequence)

    @staticmethod
    def generate_prompt(gloss_sequence): # Generate a prompt for gloss-to-text translation
        prompt = (
            f"Translate the following sequence of ASL glosses into a coherent English sentence: {gloss_sequence}. "
            f"Ensure the sentence is grammatically correct and reflects ASL's topic-comment structure where applicable. "
            f"Please use context to infer the most natural phrasing and provide only the translated English sentence without additional explanations.\n"
            f"Note: ASL often uses a topic-comment structure, omits auxiliary verbs, and emphasizes spatial relationships. "
            f"For example, glosses 'MOTHER LOVE FAMILY' might translate to 'Mother loves family' rather than a literal concatenation."
        )
        # print(f'\nGenerated prompt for {gloss_sequence}: {prompt}')
        return prompt
    
    def translate(self, gloss_sequence, max_tokens=100, temperature=0.7): # Process a sequence of glosses to produce an English sentence
        if not Gloss2TextTranslator.validate_glosses(gloss_sequence): return None
        gloss_sequence = [gloss.strip().upper() for gloss in gloss_sequence] # Standardize glosses
        prompt = Gloss2TextTranslator.generate_prompt(gloss_sequence)
        try:
            translation = self.client.chat.completions.create(model=self.model_name, messages=[
                {'role': 'system', 'content': 'You are an expert translator converting American Sign Language (ASL) glosses to natural English sentences.'},
                {'role': 'user', 'content': prompt}
            ], max_tokens=max_tokens, temperature=temperature).choices[0].message.content.strip()

            # if translation and save_output:
            #     output_path = self.output_dir / f'translation_{id(gloss_sequence)}.json'
            #     with open(output_path, 'w') as f:
            #         json.dump({'glosses': gloss_sequence, 'translation': translation}, f, indent=4)
            #         print(f'Saved translation to {output_path}')
            return translation
        except OpenAIError as e:
            print(f'OpenAI API error: {str(e)}')
            return None
        except Exception as e:
            print(f'Unexpected error: {str(e)}')
            return None

    def batch_translate(self, gloss_sequences): # Process multiple gloss sequences
        translations = []
        for glosses in tqdm(gloss_sequences):
            translations.append(self.translate(glosses))
        return translations


if __name__ == '__main__':
    gpt_translator = Gloss2TextTranslator(model_name='gpt-4o-mini')
    gloss_sequences = [
        ['ELECTION', 'QUAESTOR', 'EUROPEAN', 'PARLIAMENT', 'DEADLINE', 'FOR', 'SUBMIT', 'NOMINATION', 'SEE', 'MINUTE'],
        ['COURT', 'AUDITORSSPECIAL', 'REPORT', 'NO', '6', '2005', 'ON', 'TRAN', 'EUROPEAN', 'NETWORK', 'FOR', 'TRANSPORT', 'VOTE'],
        ['COMMUNITY', 'PARTICIPATION', 'IN', 'CAPITAL', 'INCREASE', 'EUROPEAN', 'INVESTMENT', 'FUND', 'VOTE'],
    ]
    translations = gpt_translator.batch_translate(gloss_sequences)
    for gloss_sequence, translation in zip(gloss_sequences, translations):
        print(f'Glosses: {gloss_sequence}\nTranslation: {translation}')