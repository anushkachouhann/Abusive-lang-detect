import filter from 'leo-profanity'
import { Filter } from 'bad-words'
import { RegExpMatcher, englishDataset, englishRecommendedTransformers } from 'obscenity'
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-expect-error - package ships without types
import * as profanityHindi from 'profanity-hindi'

export interface LeoProfanityAnalysisResult {
  isClean: boolean
  flaggedWords: string[]
  cleanedText: string
  message: string
}

export default class LeoProfanityService {
  private supportedLanguages = ['en', 'fr', 'ru', 'es', 'de', 'it', 'pt', 'pl', 'tr', 'ar', 'hi', 'ja', 'ko', 'zh']
  private obscenityMatcher: RegExpMatcher
  private badWordsFilter: Filter

  constructor() {
    // Initialize Leo Profanity with English dictionary
    filter.loadDictionary('en')

    // Prepare obscenity matcher for obfuscated variants (s_h_i_t, $#!t, @$$, etc.)
    const englishData = englishDataset.build()
    this.obscenityMatcher = new RegExpMatcher({
      ...englishData,
      ...englishRecommendedTransformers,
    })

    // Initialize bad-words filter (English / transliterated abuses)
    this.badWordsFilter = new Filter()
  }

  /**
   * Analyzes text using Leo Profanity + Obscenity for multilingual detection
   * and obfuscated variants.
   */
  async leoProfanityAnalyzeText(text: string, language?: string): Promise<LeoProfanityAnalysisResult> {
    try {
      const lang = language && this.supportedLanguages.includes(language) ? language : 'en'

      this.loadDictionarySafe(lang)

      // Leo-profanity check (direct dictionary)
      const leoProfane = filter.check(text)
      const cleanedText = filter.clean(text)
      const flaggedWords = this.extractFlaggedWords(text)

      // Obscenity matcher for obfuscated English variants
      const obscenityMatches = this.obscenityMatcher.getAllMatches(text)
      const obscenityWords: string[] = []

      // bad-words for English / Hinglish style abuses
      const badWordsProfane = this.badWordsFilter.isProfane(text)

      // profanity-hindi for Hindi / Hinglish abuses (via library, no custom list)
      const hindiProfane =
        typeof profanityHindi.isMessageDirty === 'function'
          ? profanityHindi.isMessageDirty(text)
          : false

      const combinedFlagged = [
        ...new Set([
          ...flaggedWords,
          ...obscenityWords,
          ...(badWordsProfane ? ['(bad-words)'] : []),
          ...(hindiProfane ? ['(profanity-hindi)'] : []),
        ]),
      ]

      /**
       * Combine all library signals: leo-profanity + obscenity + bad-words + profanity-hindi.
       * This should catch obfuscated variants like s_h_i_t, @$$, etc.
       */
      const isProfane =
        leoProfane ||
        obscenityMatches.length > 0 ||
        badWordsProfane ||
        hindiProfane ||
        combinedFlagged.length > 0

      return {
        isClean: !isProfane,
        flaggedWords: combinedFlagged,
        cleanedText,
        message: isProfane
          ? `Content contains ${combinedFlagged.length} inappropriate word(s): ${combinedFlagged.join(', ')}`
          : 'Content is clean',
      }
    } catch (error) {
      throw new Error(
        `Failed to analyze text with profanity services: ${error instanceof Error ? error.message : 'Unknown error'}`
      )
    }
  }

  /**
   * Extract flagged words from text using Leo Profanity dictionaries
   */
  private extractFlaggedWords(text: string): string[] {
    const flagged: string[] = []
    const lowerText = text.toLowerCase()

    const dictionary = filter.list()
    const words = lowerText.split(/\s+/)

    words.forEach((word) => {
      const cleanWord = word.replace(/[^\p{L}\p{N}]/gu, '')
      if (cleanWord && dictionary.includes(cleanWord)) {
        flagged.push(word)
      }
    })

    return [...new Set(flagged)]
  }

  /**
   * Load Leo Profanity dictionary safely, with Hindi add-on and English fallback.
   */
  private loadDictionarySafe(language: string) {
    try {
      filter.loadDictionary(language)
    } catch {
      filter.loadDictionary('en')
    }
  }

  /**
   * Check if text is clean (convenience method)
   */
  async leoProfanityCheckText(text: string, language?: string): Promise<LeoProfanityAnalysisResult> {
    return this.leoProfanityAnalyzeText(text, language)
  }

  /**
   * Get supported languages
   */
  getSupportedLanguages(): string[] {
    return this.supportedLanguages
  }
}
