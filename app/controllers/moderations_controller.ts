import type { HttpContext } from '@adonisjs/core/http'
import { MultipartFile } from '@adonisjs/core/bodyparser'
import NsfwDetectionService from '#services/nsfw_detection_service'

export default class ModerationsController {
  private nsfwService = new NsfwDetectionService()

  async analyze({ request, response }: HttpContext) {
    try {
      const file = request.file('file', {
        size: '50mb',
        extnames: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm']
      }) as MultipartFile

      if (!file) {
        return response.badRequest({ error: 'No file uploaded' })
      }

      if (!file.isValid) {
        return response.badRequest({ error: 'Invalid file', details: file.errors })
      }

      const result = await this.nsfwService.analyzeFile(file)

      return response.ok({
        success: true,
        data: {
          id: result.id,
          fileName: result.fileName,
          fileType: result.fileType,
          isNsfw: result.isNsfw,
          nsfwScore: result.nsfwScore,
          detectionsCount: result.detectionsCount,
          bodyCoverage: result.bodyCoverage,
          clothingAnalysis: result.clothingAnalysis,
          detailedDescription: result.detailedDescription,
          coveragePercentage: result.coveragePercentage,
          detectedGender: result.detectedGender,
          confidenceScore: result.confidenceScore,
          detailedAnalysis: result.detailedAnalysis,
          status: result.status,
          createdAt: result.createdAt
        }
      })

    } catch (error) {
      return response.internalServerError({
        success: false,
        error: 'Analysis failed',
        message: error.message
      })
    }
  }

  /**
   * Get all moderation results
   */
  async index({ request, response }: HttpContext) {
    try {
      const page = request.input('page', 1)
      const limit = request.input('limit', 20)

      const results = await this.nsfwService.getResults(page, limit)

      return response.ok({
        success: true,
        data: results.serialize()
      })

    } catch (error) {
      return response.internalServerError({
        success: false,
        error: 'Failed to fetch results',
        message: error.message
      })
    }
  }

  /**
   * Get a specific moderation result
   */
  async show({ params, response }: HttpContext) {
    try {
      const result = await this.nsfwService.getResult(params.id)

      return response.ok({
        success: true,
        data: result.serialize()
      })

    } catch (error) {
      return response.notFound({
        success: false,
        error: 'Result not found'
      })
    }
  }

  /**
   * Delete a moderation result
   */
  async destroy({ params, response }: HttpContext) {
    try {
      await this.nsfwService.deleteResult(params.id)

      return response.ok({
        success: true,
        message: 'Result deleted successfully'
      })

    } catch (error) {
      return response.notFound({
        success: false,
        error: 'Result not found'
      })
    }
  }

  /**
   * Health check for the service
   */
  async health({ response }: HttpContext) {
    try {
      // Check if ML service is running
      const fetch = (await import('node-fetch')).default
      const mlResponse = await fetch('http://localhost:8001/health')

      return response.ok({
        success: true,
        services: {
          api: 'healthy',
          mlService: mlResponse.ok ? 'healthy' : 'unhealthy'
        }
      })

    } catch (error) {
      return response.ok({
        success: true,
        services: {
          api: 'healthy',
          mlService: 'unhealthy'
        }
      })
    }
  }
}