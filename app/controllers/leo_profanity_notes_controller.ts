import type { HttpContext } from '@adonisjs/core/http'
import Note from '#models/note'
import LeoProfanityService from '#services/leo_profanity_service'
import {
  createLeoProfanityNoteValidator,
  updateLeoProfanityNoteValidator,
} from '#validators/leo_profanity_note'

export default class LeoProfanityNotesController {
  private leoProfanityService: LeoProfanityService

  constructor() {
    this.leoProfanityService = new LeoProfanityService()
  }

  async leoProfanityIndex({ response }: HttpContext) {
    try {
      const notes = await Note.query().orderBy('created_at', 'desc')
      return response.json({
        success: true,
        data: notes,
      })
    } catch (error) {
      return response.status(500).json({
        success: false,
        message: 'Failed to fetch notes',
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  async leoProfanityStore({ request, response }: HttpContext) {
    const payload = await request.validateUsing(createLeoProfanityNoteValidator)

    try {
      // Analyze title and content for profanity
      const [titleAnalysis, contentAnalysis] = await Promise.all([
        this.leoProfanityService.leoProfanityAnalyzeText(payload.title, payload.language),
        this.leoProfanityService.leoProfanityAnalyzeText(payload.content, payload.language),
      ])

      const isClean = titleAnalysis.isClean && contentAnalysis.isClean

      if (!isClean) {
        const allFlaggedWords = [
          ...titleAnalysis.flaggedWords,
          ...contentAnalysis.flaggedWords,
        ]
        const uniqueFlagged = [...new Set(allFlaggedWords)]

        return response.badRequest({
          success: false,
          message: 'Your note contains inappropriate language',
          errors: {
            flaggedWords: uniqueFlagged,
            titleAnalysis,
            contentAnalysis,
          },
        })
      }

      // Create note if content is clean
      const note = await Note.create({
        title: payload.title,
        content: payload.content,
        language: payload.language || null,
        isFlagged: false,
        flaggedWords: null,
      })

      return response.created({
        success: true,
        message: 'Note created successfully',
        data: note,
      })
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'

      return response.status(500).json({
        success: false,
        message: 'Failed to moderate content with Leo Profanity',
        error: errorMessage,
      })
    }
  }

  async leoProfanityShow({ params, response }: HttpContext) {
    try {
      const note = await Note.find(params.id)

      if (!note) {
        return response.notFound({
          success: false,
          message: 'Note not found',
        })
      }

      return response.json({
        success: true,
        data: note,
      })
    } catch (error) {
      return response.status(500).json({
        success: false,
        message: 'Failed to fetch note',
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  async leoProfanityUpdate({ params, request, response }: HttpContext) {
    try {
      const note = await Note.find(params.id)

      if (!note) {
        return response.notFound({
          success: false,
          message: 'Note not found',
        })
      }

      const payload = await request.validateUsing(updateLeoProfanityNoteValidator)

      // Only analyze provided fields
      const analyses: Array<{ analysis: any; field: keyof typeof payload; value: string }> = []

      if (payload.title) {
        const titleAnalysis = await this.leoProfanityService.leoProfanityAnalyzeText(
          payload.title,
          payload.language
        )
        analyses.push({ analysis: titleAnalysis, field: 'title', value: payload.title })
      }

      if (payload.content) {
        const contentAnalysis = await this.leoProfanityService.leoProfanityAnalyzeText(
          payload.content,
          payload.language
        )
        analyses.push({ analysis: contentAnalysis, field: 'content', value: payload.content })
      }

      // Check if any analysis failed
      const failedAnalysis = analyses.find(analysis => !analysis.analysis.isClean)

      if (failedAnalysis) {
        return response.badRequest({
          success: false,
          message: `${failedAnalysis.field} contains inappropriate language`,
          errors: {
            flaggedWords: failedAnalysis.analysis.flaggedWords,
            cleanedText: failedAnalysis.analysis.cleanedText,
            field: failedAnalysis.field,
          },
        })
      }

      // Update note with clean content
      if (payload.title) note.title = payload.title
      if (payload.content) note.content = payload.content
      if (payload.language) note.language = payload.language

      await note.save()

      return response.json({
        success: true,
        message: 'Note updated successfully (moderated by Leo Profanity)',
        data: note,
      })
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'

      return response.status(500).json({
        success: false,
        message: 'Failed to moderate content with Leo Profanity',
        error: errorMessage,
      })
    }
  }

  async leoProfanityDestroy({ params, response }: HttpContext) {
    try {
      const note = await Note.find(params.id)

      if (!note) {
        return response.notFound({
          success: false,
          message: 'Note not found',
        })
      }

      await note.delete()

      return response.json({
        success: true,
        message: 'Note deleted successfully',
      })
    } catch (error) {
      return response.status(500).json({
        success: false,
        message: 'Failed to delete note',
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  // Additional utility method for quick profanity check
  async quickProfanityCheck({ request, response }: HttpContext) {
    try {
      const { text, language } = request.only(['text', 'language'])

      if (!text) {
        return response.badRequest({
          success: false,
          message: 'Text is required'
        })
      }

      const analysis = await this.leoProfanityService.leoProfanityAnalyzeText(text, language)

      return response.json({
        success: true,
        data: analysis
      })
    } catch (error) {
      return response.status(500).json({
        success: false,
        message: 'Failed to check profanity',
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }
}