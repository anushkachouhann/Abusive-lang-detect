import type { HttpContext } from '@adonisjs/core/http'
import { MultipartFile } from '@adonisjs/core/bodyparser'
import NsfwDetectionService from '#services/nsfw_detection_service'
// import OpenVINODetectionService from '#services/openvino_detection_service'
import Moderation from '#models/moderation_result'
import { cuid } from '@adonisjs/core/helpers'
import { DateTime } from 'luxon'

export default class ModerationsController {
  private nsfwService = new NsfwDetectionService()

  /**
   * Analyze image/video with OpenVINO
   */
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

      // Save file temporarily with proper filename handling
      const fileName = file.clientName || file.fileName || 'uploaded-file.jpg'
      const uniqueId = cuid()
      const tmpPath = `./tmp/${uniqueId}-${fileName}`
      // await file.move('./tmp', { name: `${uniqueId}-${fileName}` })

      try {
        // Also run original NSFW service for comparison
        console.log('About to call NSFW service with file:', fileName)
        const nsfwResult = await this.nsfwService.analyzeFile(file)
        console.log('NSFW service completed successfully')

        // Save to database
        const moderation = await Moderation.create({
          fileName: file.fileName,
          fileType: file.extname?.replace('.', '') as 'image' | 'video' || 'image',
          filePath: tmpPath,
          isNsfw: nsfwResult.isNsfw,
          nsfwScore: nsfwResult.nsfwScore,
          detectionsCount: nsfwResult.detectionsCount,
          bodyCoverage: nsfwResult.bodyCoverage,
          clothingAnalysis: nsfwResult.clothingAnalysis,
          detailedDescription: nsfwResult.detailedDescription,
          coveragePercentage: nsfwResult.coveragePercentage,
          detectedGender: nsfwResult.detectedGender,
          confidenceScore: nsfwResult.confidenceScore,
          detailedAnalysis: nsfwResult.detailedAnalysis,
          status: 'completed',
          analysisDetails: JSON.stringify({
            // openvino: result.data?.detections || {},
            originalService: nsfwResult,
            // processingTime: result.processingTime
          }),
          createdAt: DateTime.now()
        })

        return response.ok({
          success: true,
          data: {
            id: moderation.id,
            fileName: moderation.fileName,
            fileType: moderation.fileType,
            filePath: moderation.filePath,
            isNsfw: moderation.isNsfw,
            nsfwScore: moderation.nsfwScore,
            detectionsCount: moderation.detectionsCount,
            bodyCoverage: moderation.bodyCoverage,
            clothingAnalysis: moderation.clothingAnalysis,
            detailedDescription: moderation.detailedDescription,
            coveragePercentage: moderation.coveragePercentage,
            detectedGender: moderation.detectedGender,
            confidenceScore: moderation.confidenceScore,
            detailedAnalysis: moderation.detailedAnalysis,
            status: moderation.status,
            createdAt: moderation.createdAt,
            // openvinoAnalysis: result.data?.detections,
            // processingTime: result.processingTime
          }
        })

      } finally {
        // Clean up temp file
        try {
          const fs = await import('fs/promises')
          await fs.unlink(tmpPath)
        } catch (e) {
          // Ignore cleanup errors
        }
      }

    } catch (error) {
      return response.internalServerError({
        success: false,
        error: 'Analysis failed',
        message: error instanceof Error ? error.message : 'Unknown error'
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

      const results = await Moderation.query()
        .orderBy('created_at', 'desc')
        .paginate(page, limit)

      return response.ok({
        success: true,
        data: results
      })
    } catch (error) {
      return response.internalServerError({
        success: false,
        error: 'Failed to fetch results',
        message: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  /**
   * Get a specific moderation result
   */
  async show({ params, response }: HttpContext) {
    try {
      const result = await Moderation.findOrFail(params.id)

      return response.ok({
        success: true,
        data: result
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
      const result = await Moderation.findOrFail(params.id)
      await result.delete()

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
   * Health check for services
   */
  async health({ response }: HttpContext) {
    try {
      const fetch = (await import('node-fetch')).default

      // Check OpenVINO service
      let openvinoHealth = 'unhealthy'
      try {
        const openvinoResponse = await fetch('http://localhost:8005/health')
        openvinoHealth = openvinoResponse.ok ? 'healthy' : 'unhealthy'
      } catch (e) {
        openvinoHealth = 'unhealthy'
      }

      // Check original ML service
      let mlHealth = 'unhealthy'
      try {
        const mlResponse = await fetch('http://localhost:8004/health')
        mlHealth = mlResponse.ok ? 'healthy' : 'unhealthy'
      } catch (e) {
        mlHealth = 'unhealthy'
      }

      return response.ok({
        success: true,
        services: {
          api: 'healthy',
          openvinoService: openvinoHealth,
          mlService: mlHealth
        }
      })
    } catch (error) {
      return response.ok({
        success: true,
        services: {
          api: 'healthy',
          openvinoService: 'unhealthy',
          mlService: 'unhealthy'
        }
      })
    }
  }

  // TODO: Re-enable OpenVINO methods after fixing import issues
  
  /**
   * Test OpenVINO directly via HTTP
   */
  async testOpenVINO({ request, response }: HttpContext) {
    try {
      const file = request.file('file', {
        size: '50mb',
        extnames: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
      }) as MultipartFile

      if (!file) {
        return response.badRequest({ error: 'No file uploaded' })
      }

      if (!file.isValid) {
        return response.badRequest({ error: 'Invalid file', details: file.errors })
      }

      // Save file temporarily
      const fileName = file.clientName || file.fileName || 'uploaded-file.jpg'
      const uniqueId = cuid()
      const tmpPath = `./tmp/${uniqueId}-${fileName}`
      await file.move('./tmp', { name: `${uniqueId}-${fileName}` })

      try {
        // Call OpenVINO service directly via HTTP
        const FormData = (await import('form-data')).default
        const fetch = (await import('node-fetch')).default
        const fs = await import('fs/promises')

        const formData = new FormData()
        const fileBuffer = await fs.readFile(tmpPath)
        formData.append('file', fileBuffer, { filename: fileName })

        const openvinoResponse = await fetch('http://localhost:8005/analyze-image', {
          method: 'POST',
          body: formData
        })

        if (!openvinoResponse.ok) {
          throw new Error(`OpenVINO service error: ${openvinoResponse.statusText}`)
        }

        const result = await openvinoResponse.json()

        return response.ok({
          success: true,
          source: 'OpenVINO Direct',
          data: result,
          message: 'OpenVINO analysis completed successfully'
        })

      } finally {
        const fs = await import('fs/promises')
        await fs.unlink(tmpPath).catch(() => { })
      }

    } catch (error) {
      return response.internalServerError({
        success: false,
        error: 'OpenVINO test failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }
  
  /**
   * Test OpenVINO NSFW detection
   */
  async analyzeNSFW({ request, response }: HttpContext) {
    try {
      const file = request.file('file', {
        size: '50mb',
        extnames: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
      }) as MultipartFile

      if (!file) {
        return response.badRequest({ error: 'No file uploaded' })
      }

      if (!file.isValid) {
        return response.badRequest({ error: 'Invalid file', details: file.errors })
      }

      // Save file temporarily
      const fileName = file.clientName || file.fileName || 'uploaded-file.jpg'
      const uniqueId = cuid()
      const tmpPath = `./tmp/${uniqueId}-${fileName}`
      await file.move('./tmp', { name: `${uniqueId}-${fileName}` })

      try {
        // Call OpenVINO NSFW service directly via HTTP
        const FormData = (await import('form-data')).default
        const fetch = (await import('node-fetch')).default
        const fs = await import('fs/promises')

        const formData = new FormData()
        const fileBuffer = await fs.readFile(tmpPath)
        formData.append('file', fileBuffer, { filename: fileName })

        const openvinoResponse = await fetch('http://localhost:8005/analyze-image-nsfw', {
          method: 'POST',
          body: formData
        })

        if (!openvinoResponse.ok) {
          throw new Error(`OpenVINO NSFW service error: ${openvinoResponse.statusText}`)
        }

        const result = await openvinoResponse.json()

        return response.ok({
          success: true,
          source: 'OpenVINO NSFW',
          data: result,
          message: 'OpenVINO NSFW analysis completed successfully'
        })

      } finally {
        const fs = await import('fs/promises')
        await fs.unlink(tmpPath).catch(() => { })
      }

    } catch (error) {
      return response.internalServerError({
        success: false,
        error: 'OpenVINO NSFW test failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  /*
  async analyzeOpenVINO({ request, response }: HttpContext) {
    return response.ok({
      success: false,
      error: 'OpenVINO integration temporarily disabled'
    })
  }

  async analyzeNSFW({ request, response }: HttpContext) {
    return response.ok({
      success: false,
      error: 'OpenVINO NSFW analysis temporarily disabled'
    })
  }
  */
}