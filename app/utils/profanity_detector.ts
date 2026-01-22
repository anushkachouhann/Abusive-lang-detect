import leoProfanity from 'leo-profanity'
import { RegExpMatcher, englishDataset, englishRecommendedTransformers } from 'obscenity'
import { Filter } from 'bad-words'
// @ts-expect-error - package ships without types
import * as profanityHindi from 'profanity-hindi'
import customAbuseWords from '#config/custom_abuse_words'
import { AllProfanity } from 'allprofanity'

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
    private allProfanity: AllProfanity
    private supportedLanguages: string[] = ['en', 'fr', 'ru', 'es', 'de', 'it', 'pt', 'pl', 'tr', 'ar', 'hi', 'ja', 'ko', 'zh']

    constructor() {
        const englishData = englishDataset.build()
        this.obscenityMatcher = new RegExpMatcher({
            ...englishData,
            ...englishRecommendedTransformers,
        })

        this.badWordsFilter = new Filter()

        this.allProfanity = new AllProfanity({
            algorithm: { matching: 'hybrid' },
            performance: { enableCaching: true }
        })

        this.allProfanity.loadIndianLanguages()
    }

    private escapeRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    }

    private reverseWords(text: string): string {
        return text
            .split(/(\s+)/)
            .map(part => /\s+/.test(part) ? part : part.split('').reverse().join(''))
            .join('')
    }

    private checkReversed(
        text: string,
        checker: (t: string) => boolean,
        detector?: (t: string) => { detectedWords?: string[] }
    ): { isProfane: boolean; flagged?: string[] } {
        const revText = this.reverseWords(text)
        const isProfane = checker(revText)

        if (!isProfane || !detector) {
            return { isProfane }
        }

        const result = detector(revText)
        // Flagged words are reversed → reverse them back for readable report
        const originalFlagged = result.detectedWords?.map(w => w.split('').reverse().join('')) || []

        return { isProfane, flagged: originalFlagged }
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
            const leoCleaned = leoProfanity.clean(text)
            const leoFlagged = this.extractFlaggedWords(text)

            const obscenityMatches = this.obscenityMatcher.getAllMatches(text)
            const obscenityWords = obscenityMatches.map(match => text.slice(match.startIndex, match.endIndex))

            const badWordsProfane = this.badWordsFilter.isProfane(text)

            const hindiProfane = typeof profanityHindi.isMessageDirty === 'function'
                ? profanityHindi.isMessageDirty(text)
                : false

            const allProfanityProfane = this.allProfanity.check(text)
            const allProfResult = this.allProfanity.detect(text)
            const allProfFlagged = allProfResult.detectedWords || []
            let allProfCleaned = allProfResult.cleanedText || leoCleaned

            const { isProfane: customProfane, flaggedWords: customFlagged } = this.checkCustomAbuseWords(text)

            const revLeo = this.checkReversed(text, t => leoProfanity.check(t.toLowerCase()))
            const revAllProf = this.checkReversed(
                text,
                t => this.allProfanity.check(t),
                t => this.allProfanity.detect(t)
            )
            const revFlagged = revAllProf.flagged || (revLeo.isProfane ? ['(reversed-word)'] : [])

            flaggedWords = [
                ...leoFlagged,
                ...obscenityWords,
                ...customFlagged,
                ...revFlagged,
                ...allProfFlagged,
                ...(badWordsProfane ? ['(bad-words-filter)'] : []),
                ...(hindiProfane ? ['(hindi-profanity)'] : []),
                ...(allProfanityProfane ? ['(allprofanity)'] : []),
                ...(revAllProf.isProfane || revLeo.isProfane ? ['(reversed-allprofanity/leo)'] : []),
            ].filter(Boolean)

            flaggedWords = [...new Set(flaggedWords)]

            const isProfane =
                leoProfane ||
                obscenityMatches.length > 0 ||
                badWordsProfane ||
                hindiProfane ||
                allProfanityProfane ||
                revLeo.isProfane ||
                revAllProf.isProfane ||
                customProfane ||
                flaggedWords.length > 0

            if (!allProfResult.cleanedText && (revAllProf.isProfane || revLeo.isProfane) && !isProfane) {
                allProfCleaned = leoCleaned

            }

            return {
                isClean: !isProfane,
                flaggedWords,
                cleanedText: allProfCleaned,
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

        const revText = text
            .split(/\s+/)
            .map(w => w.split('').reverse().join(''))
            .join(' ')
        if (leoProfanity.check(revText.toLowerCase())) return true

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
                (abuseWord as string)
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