from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime, timezone

from sasnl.agents.gates import mismatch_score
from sasnl.agents.llm_agents import ClaudeStructuredAgent
from sasnl.models import AgentOutput, EvidenceItem, Utterance

SEVERITY_ORDER = ["none", "mild", "moderate", "severe"]

FILLER_PHONO = {"um", "uh"}
FILLER_LEX = {"like", "well"}
FILLER_MULTI = {"you know", "i mean"}
MENTAL_STATE = {"think", "know", "believe", "remember", "forget", "wonder", "guess", "understand"}
POLITENESS = {"please", "sorry", "thanks", "thank", "could", "would"}
HEDGES = {"maybe", "kind", "sort"}
CAUSAL = {"because", "so", "therefore", "since"}
COND = {"if", "then"}
ADVERS = {"but", "however", "although"}
EMOTION_TERMS = {
    "happy",
    "sad",
    "angry",
    "upset",
    "excited",
    "afraid",
    "worried",
    "love",
    "hate",
}
IDIOMS = {"piece of cake", "break a leg", "spill the beans", "under the weather"}
LCM_DAV = {"run", "walk", "grab", "take", "go"}
LCM_IAV = {"help", "hurt", "praise", "ignore"}
LCM_SV = {"love", "hate", "admire", "fear", "trust"}


class FullBatteryRunner:
    def __init__(self, llm_client, mismatch_threshold: float = 0.7, enable_t3: bool = False):
        self.llm = llm_client
        self.mismatch_threshold = mismatch_threshold
        self.enable_t3 = enable_t3

    def run(self, context: dict) -> dict[str, AgentOutput]:
        outputs: dict[str, AgentOutput] = {}
        student = context.get("student_utterances", [])
        all_utts = context.get("all_utterances", [])
        turn_pairs = self._turn_pairs(all_utts)

        outputs["FillerWordAgent"] = self.filler_word(student)
        outputs["FalseStartSelfCorrectionAgent"] = self.false_start(student)
        outputs["RepetitionAgent"] = self.repetition_immediate(student)
        outputs["DelayedRepetitionAgent"] = self.repetition_delayed(student)
        outputs["SpeechRateRhythmAgent"] = self.speech_rate(student)
        outputs["MLUAgent"] = self.mlu(student)
        outputs["VocabularyDiversityAgent"] = self.vocab_diversity(student)
        outputs["NDWAgent"] = self.ndw(student)
        outputs["LexicalDensityAgent"] = self.lexical_density(student)
        outputs["SentenceComplexityAgent"] = self.sentence_complexity(student)
        outputs["AgreementErrorAgent"] = self.agreement_error(student)
        outputs["PronounReversalAgent"] = self.pronoun_reversal(student)
        outputs["IdiosyncraticWordAgent"] = self.idiosyncratic_words(student)
        outputs["SemanticCoherenceAgent"] = self.semantic_coherence(student, all_utts)
        outputs["GlobalCoherenceAgent"] = self.global_coherence(student)
        outputs["SentimentISLAgent"] = self.sentiment_isl(student)
        outputs["SemanticRelationshipAgent"] = self.semantic_relationship(student)
        outputs["FigurativeLanguageAgent"] = self.figurative_language(student)
        outputs["GreetingAgent"] = self.greeting(all_utts)
        outputs["DepartureAgent"] = self.departure(all_utts)
        outputs["InitiationAgent"] = self.initiation(turn_pairs)
        outputs["TurnTakingAgent"] = self.turn_taking(all_utts)
        outputs["QuestionUseAgent"] = self.question_use(student)
        outputs["InformativenessAgent"] = self.informativeness(student)
        outputs["PolitenessHedgingAgent"] = self.politeness_hedging(student)
        outputs["TopicManagementAgent"] = self.topic_management(student)
        outputs["WorkingMemoryAgent"] = self.working_memory(student)
        outputs["NarrativeMacrostructureAgent"] = self.narrative_macro(all_utts)
        outputs["CohesionDeviceAgent"] = self.cohesion(student)
        outputs["CausalReasoningAgent"] = self.causal_reasoning(student)
        outputs["OrganisationScoreAgent"] = self.organisation(student)
        outputs["ISLAgent"] = self.isl(student)
        outputs["PerspectiveTakingAgent"] = self.perspective(student)
        outputs["ReferenceTrackingAgent"] = self.reference_tracking(student)
        outputs["ContingentResponseAgent"] = self.contingent(turn_pairs)
        outputs["UtteranceIntonationAgent"] = self.intonation_proxy(student)

        if self.enable_t3:
            outputs["SemanticTangentialityAgent"] = self.semantic_tangentiality(turn_pairs)
            outputs["LCMAbstractionAgent"] = self.lcm_abstraction(student)
        else:
            outputs["SemanticTangentialityAgent"] = AgentOutput.skipped(
                "SemanticTangentialityAgent", "utterance", "student", "t3_disabled"
            )
            outputs["LCMAbstractionAgent"] = AgentOutput.skipped(
                "LCMAbstractionAgent", "utterance", "student", "t3_disabled"
            )

        outputs["SarcasmDetectionAgent"] = self.sarcasm(turn_pairs)
        outputs["EmpathyAgent"] = self.empathy(turn_pairs)
        outputs["ExecutiveFunctionAgent"] = self.executive_function(student)
        return outputs

    def _base(self, name: str, level: str, metrics: dict, interpretation: dict | None = None, evidence: list[EvidenceItem] | None = None) -> AgentOutput:
        return AgentOutput(
            agent_name=name,
            version="1.0",
            transcript_level=level,
            speaker_scope="student",
            status="completed",
            computed_at=datetime.now(timezone.utc).isoformat(),
            metrics={"source": "nlp", **metrics},
            interpretation=interpretation or {},
            evidence=evidence or [],
        )

    def _tokens(self, utterances: list[Utterance]) -> list[str]:
        return [t.text.lower() for u in utterances for t in u.tokens]

    def _turn_pairs(self, utterances: list[Utterance]) -> list[tuple[Utterance, Utterance]]:
        pairs = []
        for i in range(1, len(utterances)):
            a, b = utterances[i - 1], utterances[i]
            if a.speaker_role == "interviewer" and b.speaker_role == "student":
                pairs.append((a, b))
        return pairs

    def _sev_from_rate(self, rate: float, mild: float, moderate: float, severe: float) -> str:
        if rate >= severe:
            return "severe"
        if rate >= moderate:
            return "moderate"
        if rate >= mild:
            return "mild"
        return "none"

    def _llm_interpret(self, name: str, level: str, utterances: list[Utterance]) -> dict:
        agent = ClaudeStructuredAgent(self.llm, level=level, name=name)
        out = agent.run({"student_utterances": utterances})
        return out.interpretation

    def filler_word(self, utterances: list[Utterance]) -> AgentOutput:
        phon, lex = 0, 0
        total = 0
        ev = []
        for u in utterances:
            toks = [t.text.lower() for t in u.tokens]
            i = 0
            while i < len(toks):
                tk = toks[i]
                if tk in FILLER_PHONO:
                    phon += 1
                    total += 1
                elif tk in FILLER_LEX:
                    lex += 1
                    total += 1
                if i + 1 < len(toks):
                    bi = f"{toks[i]} {toks[i+1]}"
                    if bi in FILLER_MULTI:
                        lex += 1
                        total += 1
                i += 1
            if total:
                ev.append(EvidenceItem("token_instance", u.utterance_id, u.text, u.start_ms, u.end_ms))
        token_count = max(1, len(self._tokens(utterances)))
        rate = 100 * total / token_count
        return self._base(
            "FillerWordAgent",
            "utterance",
            {
                "total_fillers": total,
                "phonological_fillers": phon,
                "lexical_fillers": lex,
                "filler_rate_per_100_words": round(rate, 3),
            },
            interpretation={
                "source": "llm",
                "model": "rule",
                "functional_label": "fluency_load",
                "severity": self._sev_from_rate(rate, 2, 5, 8),
                "severity_scale": "none | mild | moderate | severe",
                "clinical_note": "Separated phonological and lexical fillers to prevent pragmatic confounding.",
                "confidence": 0.8,
            },
            evidence=ev,
        )

    def false_start(self, utterances: list[Utterance]) -> AgentOutput:
        hits = 0
        for u in utterances:
            low = u.text.lower()
            if any(x in low for x in ["i mean", "no wait", "actually", "sorry"]):
                hits += 1
        words = max(1, sum(u.word_count for u in utterances))
        rate = 100 * hits / words
        out = self._base(
            "FalseStartSelfCorrectionAgent",
            "utterance",
            {"false_start_count": hits, "false_start_rate_per_100_words": round(rate, 3)},
        )
        out.interpretation = self._llm_interpret("FalseStartSelfCorrectionAgent", "utterance", utterances)
        return out

    def repetition_immediate(self, utterances: list[Utterance]) -> AgentOutput:
        reps = 0
        for i in range(1, len(utterances)):
            a = utterances[i - 1].text.lower().split()
            b = utterances[i].text.lower().split()
            overlap = len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
            if overlap > 0.5:
                reps += 1
        rate = 100 * reps / max(1, len(utterances))
        return self._base("RepetitionAgent", "utterance", {"immediate_repetition_count": reps, "repetition_rate": round(rate, 3)})

    def repetition_delayed(self, utterances: list[Utterance]) -> AgentOutput:
        phrases = Counter()
        for u in utterances:
            toks = u.text.lower().split()
            for n in range(4, 7):
                for i in range(0, len(toks) - n + 1):
                    phrases[" ".join(toks[i : i + n])] += 1
        recurring = [p for p, c in phrases.items() if c > 1]
        rate = 100 * len(recurring) / max(1, len(utterances))
        return self._base("DelayedRepetitionAgent", "full_transcript", {"recurring_scripted_phrases": recurring[:20], "delayed_repetition_rate": round(rate, 3)})

    def speech_rate(self, utterances: list[Utterance]) -> AgentOutput:
        wpms = []
        lengths = []
        for u in utterances:
            dur_min = max((u.end_ms - u.start_ms) / 60000.0, 1e-3)
            wpms.append(u.word_count / dur_min)
            lengths.append(u.word_count)
        mean_wpm = statistics.fmean(wpms) if wpms else 0.0
        sdlu = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
        rlu = (max(lengths) - min(lengths)) if lengths else 0
        cv = sdlu / max(1e-6, (statistics.fmean(lengths) if lengths else 1.0))
        return self._base(
            "SpeechRateRhythmAgent",
            "utterance",
            {
                "mean_wpm": round(mean_wpm, 3),
                "sdlu": round(sdlu, 3),
                "rlu": rlu,
                "rhythm_cv": round(cv, 3),
                "clinical_flag": mean_wpm < 80 or mean_wpm > 200 or cv > 1.5,
            },
        )

    def mlu(self, utterances: list[Utterance]) -> AgentOutput:
        words = sum(u.word_count for u in utterances)
        mlu_w = words / max(1, len(utterances))
        morph = words + sum(u.text.count("ing") + u.text.count("ed") + u.text.count("'s") for u in utterances)
        mlu_m = morph / max(1, len(utterances))
        return self._base("MLUAgent", "utterance", {"mlu_w": round(mlu_w, 3), "mlu_m": round(mlu_m, 3), "utterance_count": len(utterances)})

    def vocab_diversity(self, utterances: list[Utterance]) -> AgentOutput:
        toks = self._tokens(utterances)
        ttr = len(set(toks)) / max(1, len(toks))
        # MATTR50
        w = 50
        if len(toks) < w:
            mattr = ttr
        else:
            vals = []
            for i in range(0, len(toks) - w + 1):
                seg = toks[i : i + w]
                vals.append(len(set(seg)) / max(1, len(seg)))
            mattr = statistics.fmean(vals) if vals else ttr
        return self._base("VocabularyDiversityAgent", "full_transcript", {"ttr": round(ttr, 4), "mattr50": round(mattr, 4)})

    def ndw(self, utterances: list[Utterance]) -> AgentOutput:
        toks = self._tokens(utterances)[:100]
        return self._base("NDWAgent", "full_transcript", {"ndw_100w": len(set(toks)), "sample_tokens": len(toks)})

    def lexical_density(self, utterances: list[Utterance]) -> AgentOutput:
        content = 0
        total = 0
        for u in utterances:
            for t in u.tokens:
                total += 1
                if len(t.text) > 3:
                    content += 1
        density = content / max(1, total)
        return self._base("LexicalDensityAgent", "utterance", {"lexical_density": round(density, 4), "content_word_count": content, "total_tokens": total})

    def sentence_complexity(self, utterances: list[Utterance]) -> AgentOutput:
        clauses = 0
        comp = 0
        for u in utterances:
            low = u.text.lower()
            clauses += 1 + sum(low.count(c) for c in ["because", "if", "when", "that", "which"])
            comp += low.count(",")
        cd = clauses / max(1, len(utterances))
        return self._base("SentenceComplexityAgent", "utterance", {"clause_count": clauses, "clausal_density": round(cd, 3), "compound_proxy": comp})

    def agreement_error(self, utterances: list[Utterance]) -> AgentOutput:
        # Heuristic for now: simple mismatch cues.
        err = 0
        tense = 0
        for u in utterances:
            low = u.text.lower()
            if "he go" in low or "she go" in low or "they goes" in low:
                err += 1
            if "yesterday" in low and "go" in low and "went" not in low:
                tense += 1
        total = max(1, sum(u.word_count for u in utterances))
        rate = 100 * (err + tense) / total
        return self._base("AgreementErrorAgent", "utterance", {"sv_agreement_errors": err, "tense_errors": tense, "error_rate_per_100_words": round(rate, 3), "clinical_flag": rate > 5})

    def pronoun_reversal(self, utterances: list[Utterance]) -> AgentOutput:
        reversals = 0
        counts = Counter()
        for u in utterances:
            toks = u.text.lower().split()
            counts.update([t for t in toks if t in {"i", "you", "he", "she", "they", "we"}])
            if "you am" in u.text.lower() or "i are" in u.text.lower():
                reversals += 1
        return self._base("PronounReversalAgent", "utterance", {"pronoun_counts": dict(counts), "reversal_count": reversals})

    def idiosyncratic_words(self, utterances: list[Utterance]) -> AgentOutput:
        rare = []
        for tk in self._tokens(utterances):
            if len(tk) > 11 and tk.isalpha():
                rare.append(tk)
        rate = 100 * len(rare) / max(1, len(self._tokens(utterances)))
        return self._base("IdiosyncraticWordAgent", "utterance", {"idiosyncratic_tokens": sorted(set(rare))[:30], "idiosyncratic_rate_per_100_words": round(rate, 3), "clinical_flag": rate > 5})

    def _sim(self, a: Utterance, b: Utterance) -> float:
        sa = set(a.text.lower().split())
        sb = set(b.text.lower().split())
        return len(sa & sb) / max(1, len(sa | sb))

    def semantic_coherence(self, student: list[Utterance], all_utts: list[Utterance]) -> AgentOutput:
        local = []
        for i in range(1, len(student)):
            local.append(self._sim(student[i - 1], student[i]))
        pairs = self._turn_pairs(all_utts)
        cross = [self._sim(a, b) for a, b in pairs]
        local_mean = statistics.fmean(local) if local else 0.0
        cross_mean = statistics.fmean(cross) if cross else 0.0
        return self._base("SemanticCoherenceAgent", "utterance", {"local_coherence_mean": round(local_mean, 4), "cross_speaker_coherence_mean": round(cross_mean, 4), "tangential_flag": local_mean < 0.3})

    def global_coherence(self, student: list[Utterance]) -> AgentOutput:
        sims = []
        for i in range(len(student)):
            for j in range(i + 1, len(student)):
                sims.append(self._sim(student[i], student[j]))
        return self._base("GlobalCoherenceAgent", "full_transcript", {"global_coherence_mean": round(statistics.fmean(sims), 4) if sims else 0.0, "pair_count": len(sims)})

    def sentiment_isl(self, utterances: list[Utterance]) -> AgentOutput:
        pos = 0
        neg = 0
        isl = 0
        for u in utterances:
            low = u.text.lower()
            for e in EMOTION_TERMS:
                if e in low:
                    isl += 1
            if any(w in low for w in ["good", "great", "happy", "love"]):
                pos += 1
            if any(w in low for w in ["bad", "sad", "angry", "hate"]):
                neg += 1
        return self._base("SentimentISLAgent", "utterance", {"positive_valence_count": pos, "negative_valence_count": neg, "isl_count": isl, "positive_negative_ratio": round(pos / max(1, neg), 3)})

    def semantic_relationship(self, utterances: list[Utterance]) -> AgentOutput:
        cue_hits = 0
        valid_links = 0
        for u in utterances:
            low = u.text.lower()
            if any(c in low.split() for c in (CAUSAL | COND)):
                cue_hits += 1
                if "," in low or " because " in low or " so " in low:
                    valid_links += 1
        coherence = valid_links / max(1, cue_hits)
        out = self._base(
            "SemanticRelationshipAgent",
            "utterance",
            {
                "cue_hit_count": cue_hits,
                "coherent_relationship_count": valid_links,
                "logical_coherence_ratio": round(coherence, 3),
            },
        )
        out.interpretation = self._llm_interpret("SemanticRelationshipAgent", "utterance", utterances)
        return out

    def figurative_language(self, utterances: list[Utterance]) -> AgentOutput:
        idiom_hits = 0
        simile_hits = 0
        metaphor_hits = 0
        for u in utterances:
            low = u.text.lower()
            if any(i in low for i in IDIOMS):
                idiom_hits += 1
            if " like " in f" {low} ":
                simile_hits += 1
            if " is a " in low:
                metaphor_hits += 1
        out = self._base(
            "FigurativeLanguageAgent",
            "utterance",
            {
                "idiom_hits": idiom_hits,
                "simile_hits": simile_hits,
                "metaphor_hits": metaphor_hits,
                "figurative_total": idiom_hits + simile_hits + metaphor_hits,
            },
        )
        out.interpretation = self._llm_interpret("FigurativeLanguageAgent", "utterance", utterances)
        return out

    def greeting(self, utterances: list[Utterance]) -> AgentOutput:
        first = " ".join(u.text.lower() for u in utterances[:3])
        present = any(w in first for w in ["hi", "hello", "hey", "good morning"])
        return self._base("GreetingAgent", "topic", {"greeting_present": present, "window_turns": min(3, len(utterances))})

    def departure(self, utterances: list[Utterance]) -> AgentOutput:
        last = " ".join(u.text.lower() for u in utterances[-3:])
        present = any(w in last for w in ["bye", "goodbye", "see you", "later"])
        return self._base("DepartureAgent", "topic", {"departure_present": present, "window_turns": min(3, len(utterances))})

    def initiation(self, pairs: list[tuple[Utterance, Utterance]]) -> AgentOutput:
        spontaneous = 0
        prompted = 0
        for a, b in pairs:
            if "?" in a.text:
                prompted += 1
            else:
                spontaneous += 1
        total = max(1, spontaneous + prompted)
        return self._base("InitiationAgent", "turn_pair", {"spontaneous_count": spontaneous, "prompted_count": prompted, "spontaneous_ratio": round(spontaneous / total, 3)})

    def turn_taking(self, utterances: list[Utterance]) -> AgentOutput:
        student = [u for u in utterances if u.speaker_role == "student"]
        interviewer = [u for u in utterances if u.speaker_role == "interviewer"]
        sw = sum(u.word_count for u in student)
        iw = sum(u.word_count for u in interviewer)
        per_turn = [u.word_count for u in student] or [0]
        cv = statistics.pstdev(per_turn) / max(1e-6, statistics.fmean(per_turn) or 1.0)
        ratio = sw / max(1, iw)
        return self._base("TurnTakingAgent", "full_transcript", {"student_words": sw, "interviewer_words": iw, "word_balance_ratio": round(ratio, 3), "turn_balance_cv": round(cv, 3), "clinical_flag": ratio < 0.3 or ratio > 0.7})

    def question_use(self, utterances: list[Utterance]) -> AgentOutput:
        total = 0
        initiated = 0
        responsive = 0
        follow_up = 0
        for i, u in enumerate(utterances):
            low = u.text.lower().strip()
            is_q = "?" in low or low.startswith(("what", "why", "how", "when", "where", "who", "do ", "did "))
            if not is_q:
                continue
            total += 1
            if i == 0:
                initiated += 1
                continue
            prev = utterances[i - 1].text.lower()
            if "?" in prev:
                follow_up += 1
            elif any(w in prev for w in ["tell", "say", "explain", "describe"]):
                responsive += 1
            else:
                initiated += 1
        out = self._base(
            "QuestionUseAgent",
            "utterance",
            {
                "total_questions": total,
                "initiated_questions": initiated,
                "responsive_questions": responsive,
                "follow_up_questions": follow_up,
            },
        )
        out.interpretation = self._llm_interpret("QuestionUseAgent", "utterance", utterances)
        return out

    def informativeness(self, utterances: list[Utterance]) -> AgentOutput:
        scores = []
        for u in utterances:
            wc = u.word_count
            if wc <= 3:
                scores.append(1)
            elif wc <= 8:
                scores.append(2)
            else:
                scores.append(3)
        return self._base("InformativenessAgent", "utterance", {"mean_score_1_to_3": round(statistics.fmean(scores), 3) if scores else 0.0, "score_distribution": dict(Counter(scores))})

    def politeness_hedging(self, utterances: list[Utterance]) -> AgentOutput:
        pol, hedge = 0, 0
        for u in utterances:
            low = u.text.lower().split()
            pol += sum(1 for t in low if t in POLITENESS)
            hedge += sum(1 for t in low if t in HEDGES)
            if "i think" in u.text.lower() or "kind of" in u.text.lower() or "sort of" in u.text.lower():
                hedge += 1
        n = max(1, len(utterances))
        return self._base("PolitenessHedgingAgent", "utterance", {"politeness_count": pol, "hedge_count": hedge, "politeness_rate_per_100_utterances": round(100 * pol / n, 3), "hedge_rate_per_100_utterances": round(100 * hedge / n, 3)})

    def topic_management(self, utterances: list[Utterance]) -> AgentOutput:
        if len(utterances) < 2:
            return self._base("TopicManagementAgent", "topic", {"topic_shift_count": 0, "topic_maintenance_mean": 0.0, "topic_maintenance_sd": 0.0})
        sims = [self._sim(utterances[i - 1], utterances[i]) for i in range(1, len(utterances))]
        mu = statistics.fmean(sims)
        sd = statistics.pstdev(sims) if len(sims) > 1 else 0.0
        th = mu - 1.5 * sd
        shifts = [i for i, s in enumerate(sims, start=1) if s < th]
        episodes = []
        last = 0
        for idx in shifts + [len(utterances) - 1]:
            episodes.append(max(1, idx - last))
            last = idx
        mean_ep = statistics.fmean(episodes) if episodes else float(len(utterances))
        sd_ep = statistics.pstdev(episodes) if len(episodes) > 1 else 0.0
        return self._base("TopicManagementAgent", "topic", {"topic_shift_count": len(shifts), "topic_shift_rate_per_100_utterances": round(100 * len(shifts) / max(1, len(utterances)), 3), "topic_maintenance_mean": round(mean_ep, 3), "topic_maintenance_sd": round(sd_ep, 3)})

    def working_memory(self, utterances: list[Utterance]) -> AgentOutput:
        referent_chain = 0
        chain_breaks = 0
        last_refs: set[str] = set()
        pronouns = {"he", "she", "they", "it", "him", "her", "them", "this", "that"}
        for u in utterances:
            toks = [t.text.lower() for t in u.tokens]
            refs = {t for t in toks if t in pronouns}
            if refs:
                referent_chain += 1
                if last_refs and refs.isdisjoint(last_refs):
                    chain_breaks += 1
                last_refs = refs
        coherence = 1.0 - (chain_breaks / max(1, referent_chain))
        out = self._base(
            "WorkingMemoryAgent",
            "topic",
            {
                "reference_chain_count": referent_chain,
                "chain_break_count": chain_breaks,
                "referential_coherence_ratio": round(coherence, 3),
            },
        )
        out.interpretation = self._llm_interpret("WorkingMemoryAgent", "topic", utterances)
        return out

    def narrative_macro(self, utterances: list[Utterance]) -> AgentOutput:
        txt = " ".join(u.text.lower() for u in utterances)
        elements = {
            "setting": int(any(w in txt for w in ["once", "yesterday", "there was"])),
            "initiating_event": int(any(w in txt for w in ["then", "suddenly", "started"])),
            "internal_response": int(any(w in txt for w in ["felt", "thought", "wanted"])),
            "plan": int(any(w in txt for w in ["plan", "decided", "going to"])),
            "consequence": int(any(w in txt for w in ["so", "therefore", "because"])),
            "reaction": int(any(w in txt for w in ["finally", "after that", "in the end"])),
        }
        return self._base("NarrativeMacrostructureAgent", "topic", {"story_grammar_elements": elements, "macrostructure_score": sum(elements.values())})

    def cohesion(self, utterances: list[Utterance]) -> AgentOutput:
        ref = lex = conj = 0
        for u in utterances:
            toks = u.text.lower().split()
            ref += sum(1 for t in toks if t in {"he", "she", "it", "they", "this", "that"})
            conj += sum(1 for t in toks if t in {"because", "so", "but", "and"})
            lex += len([t for t, c in Counter(toks).items() if c > 1])
        n = max(1, len(utterances))
        return self._base("CohesionDeviceAgent", "utterance", {"referential_count": ref, "lexical_count": lex, "conjunctive_count": conj, "cohesion_rate_per_session": round((ref + lex + conj) / n, 3)})

    def causal_reasoning(self, utterances: list[Utterance]) -> AgentOutput:
        c = k = a = 0
        for u in utterances:
            toks = u.text.lower().split()
            c += sum(1 for t in toks if t in CAUSAL)
            k += sum(1 for t in toks if t in COND)
            a += sum(1 for t in toks if t in ADVERS)
        words = max(1, sum(u.word_count for u in utterances))
        return self._base("CausalReasoningAgent", "utterance", {"causal_count": c, "conditional_count": k, "adversative_count": a, "connective_rate_per_100_words": round(100 * (c + k + a) / words, 3)})

    def organisation(self, utterances: list[Utterance]) -> AgentOutput:
        planning = 0
        repair = 0
        for u in utterances:
            low = u.text.lower()
            if any(x in low for x in ["first of all", "let me explain", "so what i mean"]):
                planning += 1
            if any(x in low for x in ["actually", "i mean", "wait no"]):
                repair += 1
        n = max(1, len(utterances))
        return self._base("OrganisationScoreAgent", "utterance", {"planning_marker_count": planning, "repair_marker_count": repair, "planning_ratio": round(planning / n, 3)})

    def isl(self, utterances: list[Utterance]) -> AgentOutput:
        emo = 0
        cog = 0
        first = 0
        third = 0
        for u in utterances:
            toks = u.text.lower().split()
            emo += sum(1 for t in toks if t in EMOTION_TERMS)
            for i, t in enumerate(toks):
                if t in MENTAL_STATE:
                    cog += 1
                    left = toks[max(0, i - 1)] if i > 0 else ""
                    if left in {"i", "we"}:
                        first += 1
                    elif left in {"he", "she", "they"}:
                        third += 1
        n = max(1, len(utterances))
        return self._base("ISLAgent", "utterance", {"emotion_term_rate_per_100_utterances": round(100 * emo / n, 3), "cognitive_verb_rate_per_100_utterances": round(100 * cog / n, 3), "first_person_cognitive": first, "third_person_cognitive": third})

    def perspective(self, utterances: list[Utterance]) -> AgentOutput:
        embedded = 0
        depth = 0
        for u in utterances:
            low = u.text.lower()
            if any(v in low for v in MENTAL_STATE) and "that" in low:
                embedded += 1
                depth += low.count("that")
        return self._base("PerspectiveTakingAgent", "utterance", {"embedded_clause_count": embedded, "mean_embedding_depth": round(depth / max(1, embedded), 3)})

    def reference_tracking(self, utterances: list[Utterance]) -> AgentOutput:
        refs = 0
        unresolved = 0
        for u in utterances:
            toks = u.text.lower().split()
            pron = [t for t in toks if t in {"he", "she", "they", "it", "him", "her", "them"}]
            refs += len(pron)
            if pron and len(toks) < 4:
                unresolved += len(pron)
        return self._base("ReferenceTrackingAgent", "topic", {"total_referring_expressions": refs, "unresolvable_count": unresolved, "referential_ambiguity_rate": round(unresolved / max(1, refs), 3)})

    def contingent(self, pairs: list[tuple[Utterance, Utterance]]) -> AgentOutput:
        cont = 0
        for a, b in pairs:
            sim = self._sim(a, b)
            rep = set(a.text.lower().split()) == set(b.text.lower().split())
            if sim > 0.4 and not rep:
                cont += 1
        total = max(1, len(pairs))
        return self._base("ContingentResponseAgent", "turn_pair", {"contingent_count": cont, "total_pairs": len(pairs), "contingent_rate": round(cont / total, 3)})

    def intonation_proxy(self, utterances: list[Utterance]) -> AgentOutput:
        dec = q = ex = 0
        for u in utterances:
            t = u.text.strip()
            if t.endswith("?"):
                q += 1
            elif t.endswith("!"):
                ex += 1
            else:
                dec += 1
        n = max(1, len(utterances))
        return self._base("UtteranceIntonationAgent", "utterance", {"declarative_ratio": round(dec / n, 3), "question_ratio": round(q / n, 3), "exclamative_ratio": round(ex / n, 3)})

    def semantic_tangentiality(self, pairs: list[tuple[Utterance, Utterance]]) -> AgentOutput:
        slopes = []
        for a, b in pairs:
            toks = b.text.lower().split()
            if len(toks) < 6:
                continue
            window_scores = []
            for i in range(0, len(toks) - 4):
                window = Utterance("tmp", "x", "student", 0, 1, 0, " ".join(toks[i : i + 5]), [])
                window_scores.append(self._sim(a, window))
            if len(window_scores) > 1:
                # simple slope proxy
                slopes.append((window_scores[-1] - window_scores[0]) / len(window_scores))
        mean_slope = statistics.fmean(slopes) if slopes else 0.0
        return self._base("SemanticTangentialityAgent", "utterance", {"mean_tangentiality_slope": round(mean_slope, 4), "negative_drift": mean_slope < 0})

    def lcm_abstraction(self, utterances: list[Utterance]) -> AgentOutput:
        dav = iav = sv = adj = 0
        for tk in self._tokens(utterances):
            if tk in LCM_DAV:
                dav += 1
            elif tk in LCM_IAV:
                iav += 1
            elif tk in LCM_SV:
                sv += 1
            elif tk.endswith(("ful", "ive", "ous", "able")):
                adj += 1
        total = max(1, dav + iav + sv + adj)
        abstraction = (1 * dav + 2 * iav + 3 * sv + 4 * adj) / total
        return self._base("LCMAbstractionAgent", "utterance", {"dav_count": dav, "iav_count": iav, "sv_count": sv, "adjective_count": adj, "mean_abstraction_level": round(abstraction, 3)})

    def sarcasm(self, pairs: list[tuple[Utterance, Utterance]]) -> AgentOutput:
        candidates = 0
        for _, b in pairs:
            pros = -0.6 if "!" not in b.text and "great" in b.text.lower() else 0.2
            text = 0.8 if any(w in b.text.lower() for w in ["great", "awesome", "love"]) else -0.2
            if mismatch_score(pros, text) > self.mismatch_threshold:
                candidates += 1
        out = self._base("SarcasmDetectionAgent", "turn_pair", {"mismatch_candidates": candidates, "gate_threshold": self.mismatch_threshold})
        if candidates == 0:
            out.status = "skipped"
            out.metrics["skip_reason"] = "mismatch_gate_not_met"
            return out
        out.interpretation = self._llm_interpret("SarcasmDetectionAgent", "turn_pair", [b for _, b in pairs])
        return out

    def empathy(self, pairs: list[tuple[Utterance, Utterance]]) -> AgentOutput:
        candidates = 0
        for a, b in pairs:
            if any(w in a.text.lower() for w in ["feel", "sad", "happy", "upset"]):
                pros = 0.1
                text = 0.5 if any(w in b.text.lower() for w in ["sorry", "understand", "that must"]) else -0.3
                if mismatch_score(pros, text) > self.mismatch_threshold:
                    candidates += 1
        out = self._base("EmpathyAgent", "turn_pair", {"alignment_mismatch_candidates": candidates, "gate_threshold": self.mismatch_threshold})
        if candidates == 0:
            out.status = "skipped"
            out.metrics["skip_reason"] = "mismatch_gate_not_met"
            return out
        out.interpretation = self._llm_interpret("EmpathyAgent", "turn_pair", [b for _, b in pairs])
        return out

    def executive_function(self, utterances: list[Utterance]) -> AgentOutput:
        planning_markers = 0
        repair_markers = 0
        for u in utterances:
            low = u.text.lower()
            planning_markers += sum(1 for s in ["first", "next", "finally", "let me explain"] if s in low)
            repair_markers += sum(1 for s in ["actually", "wait", "i mean"] if s in low)
        n = max(1, len(utterances))
        out = self._base("ExecutiveFunctionAgent", "topic", {"planning_ratio": round(planning_markers / n, 3), "repair_ratio": round(repair_markers / n, 3)})
        out.interpretation = self._llm_interpret("ExecutiveFunctionAgent", "topic", utterances)
        return out
