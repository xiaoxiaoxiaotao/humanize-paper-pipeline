#!/usr/bin/env python3
"""
AI Writing Pattern Detector for Academic Text

Detects common patterns that indicate AI-generated academic writing:
- Repetitive sentence structures
- Overused transition words
- Abstract placeholder language
- Low vocabulary diversity
- Mechanical paragraph patterns
"""

import re
import sys
import json
import argparse
from collections import Counter
from typing import List, Dict, Tuple
import statistics


class AIDetector:
    """Detects AI writing patterns in academic text."""
    
    # Common AI transition words that appear at sentence starts
    AI_TRANSITIONS = [
        'moreover', 'furthermore', 'additionally', 'in addition',
        'it is important to note that', 'it should be noted that',
        'it is worth noting that', 'notably', 'significantly',
        'however', 'nevertheless', 'therefore', 'thus',
        'consequently', 'hence', 'accordingly', 'overall',
        'in conclusion', 'to summarize', 'to sum up',
        'taken together', 'in essence', 'firstly', 'secondly',
        'thirdly', 'lastly', 'finally'
    ]
    
    # Abstract placeholder phrases common in AI writing
    ABSTRACT_PHRASES = [
        'various aspects', 'multiple factors', 'different perspectives',
        'in terms of', 'with regard to', 'with respect to',
        'it can be seen that', 'it has been shown that',
        'plays an important role', 'plays a crucial role',
        'serves as', 'acts as', 'functions as',
        'various ways', 'multiple dimensions', 'different angles',
        'significant impact', 'profound effect', 'important implications',
        'key factors', 'critical aspects', 'essential components',
        'a variety of', 'a number of', 'a range of',
        'in the context of', 'from the perspective of',
        'in the realm of', 'in the field of',
        'has gained attention', 'has drawn interest',
        'has been widely studied', 'has been extensively researched',
        'leveraging', 'utilizing', 'using', 'employing',
        'substantial', 'considerable', 'significant'
    ]
    
    # Hedging language common in AI writing
    HEDGING_PHRASES = [
        'may suggest', 'might indicate', 'could imply',
        'would seem', 'appears to be', 'seems to',
        'to some extent', 'in some cases', 'to a certain degree',
        'tends to', 'is likely to', 'is possible that',
        'potentially', 'presumably', 'arguably'
    ]
    
    # Passive voice markers
    PASSIVE_MARKERS = [
        'is', 'are', 'was', 'were', 'been', 'be', 'being'
    ]
    
    def __init__(self, text: str):
        """Initialize with text to analyze."""
        self.text = text
        self.paragraphs = self._split_paragraphs()
        self.sentences = self._split_sentences()
        
    def _split_paragraphs(self) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = [p.strip() for p in self.text.split('\n\n') if p.strip()]
        if not paragraphs:
            # If no double newlines, treat whole text as one paragraph
            paragraphs = [self.text.strip()]
        return paragraphs
    
    def _split_sentences(self) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter (improved regex)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', self.text)
        return [s.strip() for s in sentences if s.strip()]
    
    def analyze_sentence_uniformity(self) -> Dict:
        """Detect if sentences have uniform length (AI pattern)."""
        if len(self.sentences) < 3:
            return {'score': 0, 'details': 'Too few sentences to analyze'}
        
        word_counts = [len(s.split()) for s in self.sentences]
        avg_length = statistics.mean(word_counts)
        std_dev = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        
        variance_ratio = std_dev / avg_length if avg_length > 0 else 0

        if avg_length < 10:
            threshold_high = 0.15
            threshold_mod = 0.25
            threshold_mild = 0.40
        elif avg_length < 18:
            threshold_high = 0.20
            threshold_mod = 0.30
            threshold_mild = 0.45
        else:
            threshold_high = 0.25
            threshold_mod = 0.35
            threshold_mild = 0.50
        
        if variance_ratio < threshold_high:
            score = 0.8
            issue = 'high_uniformity'
        elif variance_ratio < threshold_mod:
            score = 0.5
            issue = 'moderate_uniformity'
        elif variance_ratio < threshold_mild:
            score = 0.2
            issue = 'mild_uniformity'
        else:
            score = 0
            issue = 'good_variation'
        
        return {
            'score': score,
            'avg_length': round(avg_length, 1),
            'std_dev': round(std_dev, 1),
            'variance_ratio': round(variance_ratio, 2),
            'issue': issue,
            'details': f'Avg sentence length: {avg_length:.1f} words, Std dev: {std_dev:.1f} (variance ratio: {variance_ratio:.2f})'
        }
    
    def detect_transition_overuse(self) -> Dict:
        """Detect overuse of mechanical AI transition words."""
        sentence_starts = [s.lower()[:50] for s in self.sentences]
        
        transition_count = 0
        found_transitions = []
        
        for start in sentence_starts:
            for trans in self.AI_TRANSITIONS:
                if start.startswith(trans):
                    transition_count += 1
                    found_transitions.append(trans)
                    break
        
        # Calculate percentage
        transition_pct = (transition_count / len(self.sentences)) * 100 if self.sentences else 0
        
        # AI typically has >20% sentences starting with these
        if transition_pct > 25:
            score = 0.9
            issue = 'excessive_transitions'
        elif transition_pct > 15:
            score = 0.6
            issue = 'high_transitions'
        elif transition_pct > 8:
            score = 0.3
            issue = 'moderate_transitions'
        else:
            score = 0
            issue = 'appropriate_transitions'
        
        return {
            'score': score,
            'count': transition_count,
            'percentage': round(transition_pct, 1),
            'found': found_transitions,
            'issue': issue,
            'details': f'{transition_count} sentences ({transition_pct:.1f}%) start with mechanical transitions'
        }
    
    def detect_abstract_language(self) -> Dict:
        """Detect overuse of abstract placeholder phrases."""
        text_lower = self.text.lower()
        
        found_phrases = []
        total_count = 0
        
        for phrase in self.ABSTRACT_PHRASES:
            count = text_lower.count(phrase)
            if count > 0:
                found_phrases.append((phrase, count))
                total_count += count
        
        # Calculate density (phrases per 100 words)
        word_count = len(self.text.split())
        density = (total_count / word_count) * 100 if word_count > 0 else 0
        
        # AI typically has density > 1.5
        if density > 2.0:
            score = 0.9
            issue = 'excessive_abstraction'
        elif density > 1.0:
            score = 0.6
            issue = 'high_abstraction'
        elif density > 0.5:
            score = 0.3
            issue = 'moderate_abstraction'
        else:
            score = 0
            issue = 'appropriate_specificity'
        
        return {
            'score': score,
            'total_count': total_count,
            'density': round(density, 2),
            'found': found_phrases,
            'issue': issue,
            'details': f'{total_count} abstract phrases found (density: {density:.2f} per 100 words)'
        }
    
    def calculate_vocabulary_diversity(self) -> Dict:
        """Calculate Type-Token Ratio (vocabulary diversity)."""
        # Tokenize and clean
        words = re.findall(r'\b[a-z]+\b', self.text.lower())
        
        if len(words) < 10:
            return {'score': 0, 'details': 'Too few words to analyze'}
        
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # AI typically has TTR < 0.45 for academic text
        if ttr < 0.40:
            score = 0.8
            issue = 'low_diversity'
        elif ttr < 0.50:
            score = 0.5
            issue = 'moderate_diversity'
        elif ttr < 0.60:
            score = 0.2
            issue = 'mild_diversity'
        else:
            score = 0
            issue = 'good_diversity'
        
        return {
            'score': score,
            'ttr': round(ttr, 3),
            'unique_words': len(unique_words),
            'total_words': len(words),
            'issue': issue,
            'details': f'Type-Token Ratio: {ttr:.3f} ({len(unique_words)} unique / {len(words)} total)'
        }
    
    def detect_passive_voice_overuse(self) -> Dict:
        """Detect excessive passive voice (common in AI academic writing)."""
        text_lower = self.text.lower()

        passive_patterns = [
            r'\b(?:is|are|was|were|been|be|being)\s+\w+ed\b',
            r'\b(?:is|are|was|were|been|be|being)\s+(?:shown|demonstrated|observed|found|noted|seen|considered|analyzed|given|taken|made|done|built|known|thought|believed|regarded|expected|required|needed|used|based|designed|developed|employed|applied|performed|conducted|established|proposed|presented|described|discussed|illustrated|implemented|evaluated|examined|investigated|measured|calculated|determined|derived|obtained|achieved|generated|produced|created|introduced|defined|identified|recognized|characterized|summarized|highlighted|emphasized|supported|validated|confirmed|verified)\b',
            r'\b(?:has|have|had)\s+been\s+\w+ed\b',
            r'\b(?:has|have|had)\s+been\s+(?:shown|demonstrated|observed|found|noted|seen|considered|analyzed|given|taken|made|done|built|known|thought|believed|regarded|expected|required|needed|used|based|designed|developed|employed|applied|performed|conducted|established|proposed|presented|described|discussed|illustrated|implemented|evaluated|examined|investigated|measured|calculated|determined|derived|obtained|achieved|generated|produced|created|introduced|defined|identified|recognized|characterized|summarized|highlighted|emphasized|supported|validated|confirmed|verified)\b',
            r'\b(?:can|could|may|might|must|shall|should|will|would)\s+be\s+\w+ed\b',
            r'\b(?:can|could|may|might|must|shall|should|will|would)\s+be\s+(?:shown|demonstrated|observed|found|noted|seen|considered|analyzed|given|taken|made|done|built|known|thought|believed|regarded|expected|required|needed|used|based|designed|developed|employed|applied|performed|conducted|established|proposed|presented|described|discussed|illustrated|implemented|evaluated|examined|investigated|measured|calculated|determined|derived|obtained|achieved|generated|produced|created|introduced|defined|identified|recognized|characterized|summarized|highlighted|emphasized|supported|validated|confirmed|verified)\b',
            r'\b(?:is|are|was|were)\s+being\s+\w+ed\b',
        ]

        passive_count = 0
        for pattern in passive_patterns:
            passive_count += len(re.findall(pattern, text_lower))

        passive_pct = (passive_count / len(self.sentences)) * 100 if self.sentences else 0

        if passive_pct > 50:
            score = 0.7
            issue = 'excessive_passive'
        elif passive_pct > 35:
            score = 0.5
            issue = 'high_passive'
        elif passive_pct > 20:
            score = 0.2
            issue = 'moderate_passive'
        else:
            score = 0
            issue = 'appropriate_voice_mix'

        return {
            'score': score,
            'count': passive_count,
            'percentage': round(passive_pct, 1),
            'issue': issue,
            'details': f'{passive_count} passive constructions detected ({passive_pct:.1f}% of sentences)'
        }
    
    def analyze_paragraph_patterns(self) -> Dict:
        """Detect repetitive paragraph opening patterns."""
        if len(self.paragraphs) < 3:
            return {'score': 0, 'details': 'Too few paragraphs to analyze'}
        
        # Extract first sentence of each paragraph
        para_starts = []
        for para in self.paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            if sentences:
                para_starts.append(sentences[0].lower()[:30])
        
        # Check for similar starts
        similar_count = 0
        for i in range(len(para_starts)):
            for j in range(i + 1, len(para_starts)):
                # Check if starts are very similar (first 20 chars)
                if para_starts[i][:20] == para_starts[j][:20]:
                    similar_count += 1
        
        similarity_ratio = similar_count / len(self.paragraphs) if self.paragraphs else 0
        
        if similarity_ratio > 0.3:
            score = 0.7
            issue = 'repetitive_openings'
        elif similarity_ratio > 0.15:
            score = 0.4
            issue = 'some_repetition'
        else:
            score = 0
            issue = 'varied_openings'
        
        return {
            'score': score,
            'similar_count': similar_count,
            'total_paragraphs': len(self.paragraphs),
            'issue': issue,
            'details': f'{similar_count} similar paragraph openings detected among {len(self.paragraphs)} paragraphs'
        }

    def analyze_burstiness(self) -> Dict:
        """Advanced burstiness analysis: sentence length variation patterns."""
        if len(self.sentences) < 3:
            return {'score': 0, 'details': 'Too few sentences'}

        word_counts = [len(s.split()) for s in self.sentences]
        if not word_counts or sum(word_counts) == 0:
            return {'score': 0, 'details': 'No words to analyze'}

        avg_len = statistics.mean(word_counts)
        std_dev = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        cv = std_dev / avg_len if avg_len > 0 else 0

        diffs = [abs(word_counts[i] - word_counts[i - 1]) for i in range(1, len(word_counts))]
        avg_diff = statistics.mean(diffs) if diffs else 0
        diff_cv = statistics.stdev(diffs) / avg_diff if len(diffs) > 1 and avg_diff > 0 else 0

        short_count = sum(1 for w in word_counts if w <= 8)
        long_count = sum(1 for w in word_counts if w >= 25)
        mix_ratio = min(short_count, long_count) / max(len(word_counts), 1)

        if avg_len < 10:
            cv_high, cv_mod = 0.12, 0.20
            diff_high = 0.25
        elif avg_len < 18:
            cv_high, cv_mod = 0.18, 0.28
            diff_high = 0.28
        else:
            cv_high, cv_mod = 0.25, 0.35
            diff_high = 0.30

        score = 0
        if cv < cv_high:
            score += 0.4
        elif cv < cv_mod:
            score += 0.2
        if diff_cv < diff_high:
            score += 0.3
        if mix_ratio < 0.1:
            score += 0.3

        return {
            'score': min(1.0, score),
            'cv': round(cv, 3),
            'diff_cv': round(diff_cv, 3),
            'mix_ratio': round(mix_ratio, 3),
            'details': f'CV={cv:.3f}, diff_CV={diff_cv:.3f}, mix_ratio={mix_ratio:.3f}'
        }

    def analyze_bigram_ttr(self) -> Dict:
        """Bigram Type-Token Ratio - AI tends to reuse the same word pairs."""
        words = re.findall(r'\b[a-z]+\b', self.text.lower())
        if len(words) < 10:
            return {'score': 0, 'details': 'Too few words'}

        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        unique_bigrams = set(bigrams)
        ttr = len(unique_bigrams) / len(bigrams) if bigrams else 1.0

        if ttr < 0.55:
            score = 0.8
        elif ttr < 0.65:
            score = 0.5
        elif ttr < 0.75:
            score = 0.2
        else:
            score = 0

        return {
            'score': score,
            'value': round(ttr, 3),
            'unique_bigrams': len(unique_bigrams),
            'total_bigrams': len(bigrams),
            'details': f'Bigram TTR: {ttr:.3f} ({len(unique_bigrams)} unique / {len(bigrams)} total)'
        }

    def analyze_clause_chain_density(self) -> Dict:
        """Detect excessive clause chaining (comma-separated clauses per sentence)."""
        if len(self.sentences) < 3:
            return {'score': 0, 'details': 'Too few sentences'}

        comma_counts = [s.count(',') + s.count(';') for s in self.sentences]
        avg_commas = statistics.mean(comma_counts) if comma_counts else 0

        if avg_commas > 4.0:
            score = 0.8
        elif avg_commas > 3.0:
            score = 0.5
        elif avg_commas > 2.0:
            score = 0.2
        else:
            score = 0

        return {
            'score': score,
            'value': round(avg_commas, 2),
            'details': f'Average {avg_commas:.1f} commas/clauses per sentence'
        }

    def analyze_sentence_opening_repetition(self) -> Dict:
        """Detect repetitive sentence opening patterns."""
        if len(self.sentences) < 5:
            return {'score': 0, 'details': 'Too few sentences'}

        openings = []
        for s in self.sentences:
            words = s.strip().split()
            if len(words) >= 2:
                openings.append(' '.join(words[:2]).lower())

        counter = Counter(openings)
        most_common = counter.most_common(1)
        if not most_common:
            return {'score': 0, 'details': 'No openings found'}

        top_pattern, top_count = most_common[0]

        if top_count >= 4:
            score = 0.8
        elif top_count >= 3:
            score = 0.5
        elif top_count >= 2:
            score = 0.2
        else:
            score = 0

        return {
            'score': score,
            'count': top_count,
            'top_pattern': top_pattern,
            'details': f'"{top_pattern}" appears {top_count} times as sentence opening'
        }

    def analyze_punctuation_density(self) -> Dict:
        """Analyze punctuation patterns - AI tends to have uniform punctuation."""
        if len(self.sentences) < 3:
            return {'score': 0, 'details': 'Too few sentences'}

        comma_count = self.text.count(',')
        period_count = self.text.count('.') + self.text.count('!') + self.text.count('?')
        ratio = comma_count / period_count if period_count > 0 else 0

        if ratio > 4.0:
            score = 0.7
        elif ratio > 3.0:
            score = 0.4
        elif ratio > 2.0:
            score = 0.15
        else:
            score = 0

        return {
            'score': score,
            'comma_period_ratio': round(ratio, 2),
            'details': f'Comma/period ratio: {ratio:.2f}'
        }

    def detect_concluding_formula(self) -> Dict:
        """Detect formulaic concluding phrases at paragraph ends."""
        concluding_patterns = [
            r'in conclusion', r'to summarize', r'in summary',
            r'overall', r'taken together', r'in essence',
            r'this paper (?:has|will have|aims to|seeks to)',
            r'the present study', r'this research', r'this work',
            r'it is (?:clear|evident|apparent) that',
            r'as (?:has been|discussed|demonstrated|shown)',
        ]

        count = 0
        text_lower = self.text.lower()
        for pattern in concluding_patterns:
            matches = re.findall(pattern, text_lower)
            count += len(matches)

        if count >= 3:
            score = 0.7
        elif count >= 2:
            score = 0.4
        elif count >= 1:
            score = 0.15
        else:
            score = 0

        return {
            'score': score,
            'count': count,
            'details': f'{count} formulaic concluding phrases detected'
        }

    def detect_hedging_overuse(self) -> Dict:
        """Detect excessive hedging language (AI hallmark)."""
        hedging_patterns = [
            r'\b(?:may|might|could|can|would)\s+(?:suggest|indicate|imply|demonstrate|show|reveal|reflect)\b',
            r'\b(?:it\s+)?(?:seems|appears)\s+(?:that|to\s+be)\b',
            r'\b(?:potentially|possibly|presumably|arguably)\b',
            r'\b(?:to\s+some\s+extent|to\s+a\s+certain\s+degree|in\s+some\s+cases)\b',
            r'\b(?:tends?\s+to|is\s+likely\s+to|may\s+be)\b',
            r'\b(?:generally|typically|usually|often)\s+(?:considered|regarded|viewed|seen)\b',
        ]

        count = 0
        text_lower = self.text.lower()
        for pattern in hedging_patterns:
            count += len(re.findall(pattern, text_lower))

        if count >= 5:
            score = 0.8
        elif count >= 3:
            score = 0.5
        elif count >= 1:
            score = 0.2
        else:
            score = 0

        return {
            'score': score,
            'count': count,
            'details': f'{count} hedging constructions detected'
        }

    def detect_definition_pattern(self) -> Dict:
        """Detect AI's tendency to use 'X is Y' definition patterns."""
        pattern = r'\b(\w+(?:\s+\w+){0,3})\s+(?:is|are)\s+(?:a|an|the)\s+(\w+(?:\s+\w+){0,5})'
        matches = re.findall(pattern, self.text.lower())
        count = len(matches)

        if count >= 5:
            score = 0.7
        elif count >= 3:
            score = 0.4
        elif count >= 1:
            score = 0.15
        else:
            score = 0

        return {
            'score': score,
            'count': count,
            'details': f'{count} "X is a Y" definition patterns detected'
        }

    def analyze_citation_distribution(self) -> Dict:
        """Analyze citation placement - AI puts all citations at sentence end."""
        citation_pattern = r'\[\d+(?:,\s*\d+)*\]|\(\w+\s+\d{4}\)'
        citations = list(re.finditer(citation_pattern, self.text))
        total = len(citations)

        if total < 3:
            return {'score': 0, 'details': 'Too few citations'}

        end_citations = 0
        for s in self.sentences:
            s_citations = list(re.finditer(citation_pattern, s))
            for m in s_citations:
                remaining = s[m.end():].strip()
                if not remaining or remaining in ['.', '!', '?']:
                    end_citations += 1

        end_ratio = end_citations / total if total > 0 else 0

        if end_ratio > 0.8:
            score = 0.7
        elif end_ratio > 0.6:
            score = 0.4
        elif end_ratio > 0.4:
            score = 0.15
        else:
            score = 0

        return {
            'score': score,
            'total_citations': total,
            'end_citation_ratio': round(end_ratio, 2),
            'details': f'{total} citations, {end_ratio*100:.0f}% at sentence end'
        }

    def analyze_info_density_uniformity(self) -> Dict:
        """Detect uniform information density across sentences."""
        if len(self.sentences) < 3:
            return {'score': 0, 'details': 'Too few sentences'}

        densities = []
        for s in self.sentences:
            content_words = len(re.findall(r'\b[a-z]{3,}\b', s.lower()))
            total_words = len(s.split())
            if total_words > 0:
                densities.append(content_words / total_words)

        if len(densities) < 3:
            return {'score': 0, 'details': 'Too few density values'}

        density_std = statistics.stdev(densities) if len(densities) > 1 else 0

        if density_std < 0.05:
            score = 0.7
        elif density_std < 0.08:
            score = 0.4
        elif density_std < 0.12:
            score = 0.15
        else:
            score = 0

        return {
            'score': score,
            'std': round(density_std, 4),
            'details': f'Info density std: {density_std:.4f}'
        }

    def analyze_paragraph_template(self) -> Dict:
        """Detect template paragraph structure (background->problem->significance->approach)."""
        if len(self.paragraphs) < 2:
            return {'score': 0, 'details': 'Too few paragraphs'}

        markers = {
            'background': [r'\b(?:in\s+recent\s+years|recently|with\s+the\s+development|traditionally)\b'],
            'problem': [r'\b(?:however|nevertheless|challenge|limitation|issue|problem|gap)\b'],
            'significance': [r'\b(?:important|significant|crucial|critical|essential|vital)\b'],
            'approach': [r'\b(?:this\s+(?:paper|study|work|research)|we\s+(?:propose|present|introduce))\b'],
        }

        marker_count = 0
        text_lower = self.text.lower()
        for category, patterns in markers.items():
            for p in patterns:
                if re.search(p, text_lower):
                    marker_count += 1
                    break

        if marker_count >= 3:
            score = 0.7
        elif marker_count >= 2:
            score = 0.35
        else:
            score = 0

        return {
            'score': score,
            'marker_count': marker_count,
            'details': f'{marker_count}/4 template markers detected'
        }

    def analyze_word_repetition(self) -> Dict:
        """Detect excessive word repetition within short spans."""
        words = re.findall(r'\b[a-z]{4,}\b', self.text.lower())
        if len(words) < 20:
            return {'score': 0, 'details': 'Too few words'}

        counter = Counter(words)
        total = len(words)
        repeated = sum(c for w, c in counter.items() if c >= 3)
        repeat_ratio = repeated / total if total > 0 else 0

        if repeat_ratio > 0.15:
            score = 0.7
        elif repeat_ratio > 0.10:
            score = 0.4
        elif repeat_ratio > 0.05:
            score = 0.15
        else:
            score = 0

        return {
            'score': score,
            'repeat_ratio': round(repeat_ratio, 3),
            'details': f'{repeat_ratio*100:.1f}% of words repeated 3+ times'
        }

    def analyze_sliding_window_ttr(self) -> Dict:
        """Sliding window TTR - detects local vocabulary repetition."""
        words = re.findall(r'\b[a-z]+\b', self.text.lower())
        if len(words) < 30:
            return {'score': 0, 'details': 'Too few words'}

        window_size = min(50, len(words) // 2)
        if window_size < 10:
            return {'score': 0, 'details': 'Text too short for sliding window'}

        ttrs = []
        for i in range(0, len(words) - window_size + 1, max(1, window_size // 4)):
            window = words[i:i + window_size]
            ttr = len(set(window)) / len(window)
            ttrs.append(ttr)

        avg_ttr = statistics.mean(ttrs) if ttrs else 0
        ttr_std = statistics.stdev(ttrs) if len(ttrs) > 1 else 0

        score = 0
        if avg_ttr < 0.55:
            score += 0.4
        elif avg_ttr < 0.65:
            score += 0.2
        if ttr_std < 0.03:
            score += 0.3
        elif ttr_std < 0.05:
            score += 0.15

        return {
            'score': min(1.0, score),
            'avg_ttr': round(avg_ttr, 3),
            'ttr_std': round(ttr_std, 4),
            'windows': len(ttrs),
            'details': f'Sliding window avg TTR: {avg_ttr:.3f}, std: {ttr_std:.4f}'
        }

    def calculate_overall_score(self, metrics: Dict) -> float:
        """Calculate overall AI probability score (0-1) using additive model."""
        score = 0.0

        score += metrics.get('sentence_uniformity', {}).get('score', 0) * 0.18
        score += metrics.get('transition_overuse', {}).get('score', 0) * 0.30
        score += metrics.get('abstract_language', {}).get('score', 0) * 0.28
        score += metrics.get('vocabulary_diversity', {}).get('score', 0) * 0.15
        score += metrics.get('passive_voice', {}).get('score', 0) * 0.12
        score += metrics.get('paragraph_patterns', {}).get('score', 0) * 0.10
        score += metrics.get('burstiness', {}).get('score', 0) * 0.10
        score += metrics.get('bigram_ttr', {}).get('score', 0) * 0.15
        score += metrics.get('clause_chain_density', {}).get('score', 0) * 0.08
        score += metrics.get('sentence_opening_repetition', {}).get('score', 0) * 0.12
        score += metrics.get('punctuation_density', {}).get('score', 0) * 0.06
        score += metrics.get('concluding_formula', {}).get('score', 0) * 0.15
        score += metrics.get('hedging_overuse', {}).get('score', 0) * 0.18
        score += metrics.get('definition_pattern', {}).get('score', 0) * 0.10
        score += metrics.get('citation_distribution', {}).get('score', 0) * 0.06
        score += metrics.get('info_density_uniformity', {}).get('score', 0) * 0.05
        score += metrics.get('paragraph_template', {}).get('score', 0) * 0.10
        score += metrics.get('word_repetition', {}).get('score', 0) * 0.10
        score += metrics.get('sliding_window_ttr', {}).get('score', 0) * 0.06

        return min(1.0, score)

    def analyze(self) -> Dict:
        """Run full analysis and return results."""
        metrics = {
            'sentence_uniformity': self.analyze_sentence_uniformity(),
            'transition_overuse': self.detect_transition_overuse(),
            'abstract_language': self.detect_abstract_language(),
            'vocabulary_diversity': self.calculate_vocabulary_diversity(),
            'passive_voice': self.detect_passive_voice_overuse(),
            'paragraph_patterns': self.analyze_paragraph_patterns(),
            'burstiness': self.analyze_burstiness(),
            'bigram_ttr': self.analyze_bigram_ttr(),
            'clause_chain_density': self.analyze_clause_chain_density(),
            'sentence_opening_repetition': self.analyze_sentence_opening_repetition(),
            'punctuation_density': self.analyze_punctuation_density(),
            'concluding_formula': self.detect_concluding_formula(),
            'hedging_overuse': self.detect_hedging_overuse(),
            'definition_pattern': self.detect_definition_pattern(),
            'citation_distribution': self.analyze_citation_distribution(),
            'info_density_uniformity': self.analyze_info_density_uniformity(),
            'paragraph_template': self.analyze_paragraph_template(),
            'word_repetition': self.analyze_word_repetition(),
            'sliding_window_ttr': self.analyze_sliding_window_ttr(),
        }

        overall_score = self.calculate_overall_score(metrics)

        if overall_score > 0.7:
            probability = 'Very High'
            recommendation = 'Text shows strong AI patterns. Significant rewriting recommended.'
        elif overall_score > 0.5:
            probability = 'High'
            recommendation = 'Text shows multiple AI patterns. Rewriting recommended.'
        elif overall_score > 0.35:
            probability = 'Moderate'
            recommendation = 'Text shows some AI patterns. Selective rewriting recommended.'
        else:
            probability = 'Low'
            recommendation = 'Text appears relatively natural. Minor adjustments may help.'

        return {
            'overall_score': round(overall_score, 3),
            'probability': probability,
            'recommendation': recommendation,
            'metrics': metrics,
            'text_stats': {
                'paragraphs': len(self.paragraphs),
                'sentences': len(self.sentences),
                'words': len(self.text.split())
            }
        }
    
    def format_report(self, results: Dict, detailed: bool = False) -> str:
        """Format analysis results as readable report."""
        report = []
        report.append("=" * 70)
        report.append("AI WRITING PATTERN DETECTION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overall assessment
        report.append(f"Overall AI Probability: {results['probability']} ({results['overall_score']:.1%})")
        report.append(f"Recommendation: {results['recommendation']}")
        report.append("")
        
        # Text statistics
        stats = results['text_stats']
        report.append(f"Text Statistics:")
        report.append(f"  - Paragraphs: {stats['paragraphs']}")
        report.append(f"  - Sentences: {stats['sentences']}")
        report.append(f"  - Words: {stats['words']}")
        report.append("")
        
        # Individual metrics
        report.append("Detailed Analysis:")
        report.append("-" * 70)
        
        metrics = results['metrics']
        
        # Sentence Uniformity
        m = metrics['sentence_uniformity']
        report.append(f"\n1. Sentence Uniformity: {self._score_indicator(m['score'])}")
        report.append(f"   {m['details']}")
        if detailed and m['score'] > 0.4:
            report.append(f"   → Issue: Sentences are too uniform in length")
            report.append(f"   → Fix: Mix short (5-10), medium (15-20), and long (25-35) word sentences")
        
        # Transition Overuse
        m = metrics['transition_overuse']
        report.append(f"\n2. Mechanical Transitions: {self._score_indicator(m['score'])}")
        report.append(f"   {m['details']}")
        if detailed and m['found']:
            report.append(f"   → Found transitions: {', '.join(set(m['found']))}")
            report.append(f"   → Fix: Replace with implicit connections or varied transitions")
        
        # Abstract Language
        m = metrics['abstract_language']
        report.append(f"\n3. Abstract Language: {self._score_indicator(m['score'])}")
        report.append(f"   {m['details']}")
        if detailed and m['found']:
            top_phrases = sorted(m['found'], key=lambda x: x[1], reverse=True)[:5]
            report.append(f"   → Most frequent: {', '.join([f'{p[0]} ({p[1]}x)' for p in top_phrases])}")
            report.append(f"   → Fix: Replace with specific concepts, named theories, concrete examples")
        
        # Vocabulary Diversity
        m = metrics['vocabulary_diversity']
        report.append(f"\n4. Vocabulary Diversity: {self._score_indicator(m['score'])}")
        report.append(f"   {m['details']}")
        if detailed and m['score'] > 0.5:
            report.append(f"   → Issue: Low vocabulary variety")
            report.append(f"   → Fix: Use more varied terminology, avoid word repetition")
        
        # Passive Voice
        m = metrics['passive_voice']
        report.append(f"\n5. Passive Voice Usage: {self._score_indicator(m['score'])}")
        report.append(f"   {m['details']}")
        if detailed and m['score'] > 0.4:
            report.append(f"   → Issue: Excessive passive constructions")
            report.append(f"   → Fix: Mix with active voice where appropriate")
        
        # Paragraph Patterns
        m = metrics['paragraph_patterns']
        report.append(f"\n6. Paragraph Opening Patterns: {self._score_indicator(m['score'])}")
        report.append(f"   {m['details']}")
        if detailed and m['score'] > 0.4:
            report.append(f"   → Issue: Repetitive paragraph openings")
            report.append(f"   → Fix: Vary how paragraphs begin")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def _score_indicator(self, score: float) -> str:
        """Convert score to visual indicator."""
        if score > 0.7:
            return "🔴 HIGH CONCERN"
        elif score > 0.4:
            return "🟡 MODERATE CONCERN"
        else:
            return "🟢 OK"


def main():
    """Command-line interface."""
    # Set UTF-8 encoding for Windows console
    import sys
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    parser = argparse.ArgumentParser(
        description='Detect AI writing patterns in academic text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_detector.py input.txt
  python ai_detector.py input.txt --detailed
  python ai_detector.py input.txt --json > results.json
        """
    )
    
    parser.add_argument('input_file', help='Text file to analyze')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed analysis with fix suggestions')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not text.strip():
        print("Error: Input file is empty", file=sys.stderr)
        sys.exit(1)
    
    # Run analysis
    detector = AIDetector(text)
    results = detector.analyze()
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(detector.format_report(results, detailed=args.detailed))


if __name__ == '__main__':
    main()
