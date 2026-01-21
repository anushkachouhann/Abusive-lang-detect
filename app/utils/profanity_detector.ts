import leoProfanity from 'leo-profanity'
import { RegExpMatcher, englishDataset, englishRecommendedTransformers } from 'obscenity'
import { Filter } from 'bad-words'
// @ts-expect-error - package ships without types
import * as profanityHindi from 'profanity-hindi'
import customAbuseWords from '#config/custom_abuse_words'

leoProfanity.loadDictionary()

interface ProfanityAnalysisResult {
    isClean: boolean
    flaggedWords: string[]
    cleanedText: string
    message: string
}

interface LeetSpeakMap {
    [key: string]: string[]
}

const CHAR_MAP: LeetSpeakMap = {
    a: ['a', '@', '4'],
    b: ['b', '8'],
    c: ['c', '('],
    d: ['d'],
    e: ['e', '3'],
    f: ['f', 'ph'],
    g: ['g', '9'],
    h: ['h', '#', '|-|'],
    i: ['i', '1', '!', 'l'],
    k: ['k', 'x'],
    l: ['l', '1', '|'],
    n: ['n'],
    o: ['o', '0'],
    s: ['s', '5', '$'],
    t: ['t', '7'],
    u: ['u', 'v'],
    v: ['v', 'u'],
    w: ['w', 'vv'],
    x: ['x'],
    y: ['y'],
    z: ['z', '2'],
}

const HINDI_CHAR_MAP: LeetSpeakMap = {
    क: ['क', 'क़'],
    ख: ['ख', 'ख़'],
    ग: ['ग', 'ग़'],
    ज: ['ज', 'ज़'],
    फ: ['फ', 'फ़'],
    ड: ['ड', 'ड़'],
    ढ: ['ढ', 'ढ़'],
    च: ['च', 'छ'],
}

class ProfanityDetector {
    private obscenityMatcher: RegExpMatcher
    private badWordsFilter: Filter
    private supportedLanguages: string[] = ['en', 'fr', 'ru', 'es', 'de', 'it', 'pt', 'pl', 'tr', 'ar', 'hi', 'ja', 'ko', 'zh']

    constructor() {
        // Initialize obscenity matcher for English
        const englishData = englishDataset.build()
        this.obscenityMatcher = new RegExpMatcher({
            ...englishData,
            ...englishRecommendedTransformers,
        })

        this.badWordsFilter = new Filter()
    }

    private escapeRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    }

    private normalizeText(text: string): string {
        return text
            .toLowerCase()
            .normalize('NFD')
            .replace(/[\u0300-\u036f]/g, '')
            .replace(/[^a-z0-9\u0900-\u097F]/g, '')
            .replace(/(.)\1{2,}/g, '$1$1')
    }

    private createFlexiblePattern(word: string): RegExp {
        const isHindi = /[\u0900-\u097F]/.test(word)

        if (isHindi) {
            const pattern = word
                .split('')
                .map((char) => {
                    const vars = HINDI_CHAR_MAP[char] || [char]
                    const escaped = vars.map(this.escapeRegex).join('|')
                    return `(?:${escaped})+[\\s\\W]*`
                })
                .join('')
            return new RegExp(pattern, 'u')
        }

        const pattern = word
            .toLowerCase()
            .split('')
            .map((char) => {
                const vars = CHAR_MAP[char] || [char]
                const escaped = vars.map(this.escapeRegex).join('|')
                return `(?:${escaped})+[\\s\\W]*`
            })
            .join('')

        return new RegExp(pattern, 'i')
    }

    private extractFlaggedWords(text: string): string[] {
        const flagged: string[] = []
        const lowerText = text.toLowerCase()
        const words = lowerText.split(/\s+/)

        const dictionary = leoProfanity.list()
        words.forEach((word) => {
            const cleanWord = word.replace(/[^\p{L}\p{N}]/gu, '')
            if (cleanWord && dictionary.includes(cleanWord)) {
                flagged.push(word)
            }
        })

        return [...new Set(flagged)]
    }

    private checkCustomAbuseWords(text: string): { isProfane: boolean; flaggedWords: string[] } {
        if (!text || text.trim().length === 0) return { isProfane: false, flaggedWords: [] }

        const lowerText = text.toLowerCase()
        const allWords = [...new Set([...customAbuseWords])]

        const flagged: string[] = []

        for (const abuseWord of allWords) {
            if (!abuseWord) continue

            const normalizedAbuse = this.normalizeText(abuseWord)

            if (normalizedAbuse.length < 3) continue

            if (/[\u0900-\u097F]/.test(abuseWord)) {
                if (text.includes(abuseWord)) {
                    flagged.push(abuseWord)
                    continue
                }
            }

            const pattern = this.createFlexiblePattern(abuseWord)
            if (pattern.test(lowerText)) {
                flagged.push(abuseWord)
            }
        }

        return {
            isProfane: flagged.length > 0,
            flaggedWords: flagged
        }
    }

    private loadDictionarySafe(language: string): void {
        try {
            leoProfanity.loadDictionary(language)
        } catch {
            leoProfanity.loadDictionary('en')
        }
    }

    async analyzeText(text: string, language?: string): Promise<ProfanityAnalysisResult> {
        try {
            const lang = language && this.supportedLanguages.includes(language) ? language : 'en'

            this.loadDictionarySafe(lang)

            const lowerText = text.toLowerCase()
            let flaggedWords: string[] = []

            const leoProfane = leoProfanity.check(lowerText)
            const cleanedText = leoProfanity.clean(text)
            const leoFlagged = this.extractFlaggedWords(text)

            const obscenityMatches = this.obscenityMatcher.getAllMatches(text)
            const obscenityWords = obscenityMatches.map(match => match.matchedText)

            const badWordsProfane = this.badWordsFilter.isProfane(text)

            const hindiProfane = typeof profanityHindi.isMessageDirty === 'function'
                ? profanityHindi.isMessageDirty(text)
                : false

            const { isProfane: customProfane, flaggedWords: customFlagged } = this.checkCustomAbuseWords(text)

            flaggedWords = [
                ...leoFlagged,
                ...obscenityWords,
                ...customFlagged,
                ...(badWordsProfane ? ['(bad-words-filter)'] : []),
                ...(hindiProfane ? ['(hindi-profanity)'] : []),
            ].filter(Boolean)

            flaggedWords = [...new Set(flaggedWords)]

            const isProfane =
                leoProfane ||
                obscenityMatches.length > 0 ||
                badWordsProfane ||
                hindiProfane ||
                customProfane ||
                flaggedWords.length > 0

            return {
                isClean: !isProfane,
                flaggedWords,
                cleanedText,
                message: isProfane
                    ? `Content contains ${flaggedWords.length} inappropriate word(s): ${flaggedWords.join(', ')}`
                    : 'Content is clean',
            }
        } catch (error) {
            throw new Error(
                `Failed to analyze text with profanity services: ${error instanceof Error ? error.message : 'Unknown error'}`
            )
        }
    }

    getSupportedLanguages(): string[] {
        return this.supportedLanguages
    }

    static containsAbuse(text: string): boolean {
        if (!text || text.trim().length === 0) return false

        const lowerText = text.toLowerCase()

        if (leoProfanity.check(lowerText)) return true

        const allWords = [...new Set([...customAbuseWords])]

        for (const abuseWord of allWords) {
            if (!abuseWord) continue

            const normalizedAbuse = text
                .toLowerCase()
                .normalize('NFD')
                .replace(/[\u0300-\u036f]/g, '')
                .replace(/[^a-z0-9\u0900-\u097F]/g, '')
                .replace(/(.)\1{2,}/g, '$1$1')

            if (normalizedAbuse.length < 3) continue

            if (/[\u0900-\u097F]/.test(abuseWord)) {
                if (text.includes(abuseWord)) return true
                continue
            }

            const pattern = new RegExp(
                abuseWord
                    .toLowerCase()
                    .split('')
                    .map((char) => {
                        const vars = CHAR_MAP[char] || [char]
                        const escaped = vars.map(char => char.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')
                        return `(?:${escaped})+[\\s\\W]*`
                    })
                    .join(''),
                'i'
            )

            if (pattern.test(lowerText)) return true
        }

        return false
    }
}

export default ProfanityDetector
export type { ProfanityAnalysisResult }