import ProfanityDetector, { ProfanityAnalysisResult } from '../utils/profanity_detector.js'

export default class LeoProfanityService {
  private profanityDetector: ProfanityDetector

  constructor() {
    this.profanityDetector = new ProfanityDetector()
  }

  async leoProfanityAnalyzeText(text: string, language?: string): Promise<ProfanityAnalysisResult> {
    return this.profanityDetector.analyzeText(text, language)
  }

  async leoProfanityCheckText(text: string, language?: string): Promise<ProfanityAnalysisResult> {
    return this.leoProfanityAnalyzeText(text, language)
  }

  static containsAbuse(text: string): boolean {
    return ProfanityDetector.containsAbuse(text)
  }

  getSupportedLanguages(): string[] {
    return this.profanityDetector.getSupportedLanguages()
  }
}