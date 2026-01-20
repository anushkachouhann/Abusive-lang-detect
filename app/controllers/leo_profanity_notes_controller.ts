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
    const notes = await Note.query().orderBy('created_at', 'desc')
    return response.json({
      success: true,
      data: notes,
    })
  }
 
  async leoProfanityStore({ request, response }: HttpContext) {
    const payload = await request.validateUsing(createLeoProfanityNoteValidator)

    try { 
      const titleAnalysis = await this.leoProfanityService.leoProfanityAnalyzeText(
        payload.title,
        payload.language
      )
      const contentAnalysis = await this.leoProfanityService.leoProfanityAnalyzeText(
        payload.content,
        payload.language
      )
 
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
            titleAnalysis: {
              isClean: titleAnalysis.isClean,
              flaggedWords: titleAnalysis.flaggedWords,
              cleanedText: titleAnalysis.cleanedText,
            },
            contentAnalysis: {
              isClean: contentAnalysis.isClean,
              flaggedWords: contentAnalysis.flaggedWords,
              cleanedText: contentAnalysis.cleanedText,
            },
          },
        })
      }
 
      const note = await Note.create({
        title: payload.title,
        content: payload.content,
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
  }
 
  async leoProfanityUpdate({ params, request, response }: HttpContext) {
    const note = await Note.find(params.id)

    if (!note) {
      return response.notFound({
        success: false,
        message: 'Note not found',
      })
    }

    const payload = await request.validateUsing(updateLeoProfanityNoteValidator)

    try { 
      if (payload.title) {
        const titleAnalysis = await this.leoProfanityService.leoProfanityAnalyzeText(
          payload.title,
          payload.language
        )
        if (!titleAnalysis.isClean) {
          return response.badRequest({
            success: false,
            message: 'Title contains inappropriate language',
            errors: {
              flaggedWords: titleAnalysis.flaggedWords,
              cleanedText: titleAnalysis.cleanedText,
            },
          })
        }
        note.title = payload.title
      }
 
      if (payload.content) {
        const contentAnalysis = await this.leoProfanityService.leoProfanityAnalyzeText(
          payload.content,
          payload.language
        )
        if (!contentAnalysis.isClean) {
          return response.badRequest({
            success: false,
            message: 'Content contains inappropriate language',
            errors: {
              flaggedWords: contentAnalysis.flaggedWords,
              cleanedText: contentAnalysis.cleanedText,
            },
          })
        }
        note.content = payload.content
      }

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
  }
}
