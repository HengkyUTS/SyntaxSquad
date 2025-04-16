import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score.rouge_scorer import RougeScorer
from tqdm import tqdm
nltk.download('punkt_tab')


class TranslationEvaluator:
    def __init__(self):
        self.rouge = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    def evaluate(self, reference, hypothesis): # Evaluate a single translation against a reference
        if not reference or not hypothesis:
            print('Empty hypothesis or reference provided')
            return {'bleu': 0.0, 'rouge1': 0.0, 'rougeL': 0.0}
        
        # BLEU score
        ref_tokens = [nltk.word_tokenize(reference.lower())]
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        bleu_score = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        
        # ROUGE score
        rouge_scores = self.rouge.score(reference.lower(), hypothesis.lower())
        return {
            'bleu': bleu_score,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure
        }
    
    def batch_evaluate(self, references, hypotheses): # Evaluate multiple translations
        if len(references) != len(references): raise ValueError('Number of references and hypotheses must be the same')
        bleu_scores, rouge1_scores, rougeL_scores = [], [], []
        for ref, hyp in tqdm(zip(references, hypotheses)):
            scores = self.evaluate(ref, hyp)
            bleu_scores.append(scores['bleu'])
            rouge1_scores.append(scores['rouge1'])
            rougeL_scores.append(scores['rougeL'])

        return {
            'bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
            'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
        }


if __name__ == '__main__':
    evaluator = TranslationEvaluator()
    references = [
        'Mother loves her family.',
        'The father works diligently.',
        'Friends support the community.',
        'The quick brown fox jumps over the lazy dog.'
    ]
    hypotheses = [
        'The mother loves the family.',
        'Father works hard.',
        'Friend helps community.',
        'The quick brown dog jumps on the log.'
    ]
    print(f'Average Scores: {evaluator.batch_evaluate(references, hypotheses)}')