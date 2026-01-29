import { MultipartFile } from '@adonisjs/core/bodyparser'
import ModerationResult from '#models/moderation_result'
import app from '@adonisjs/core/services/app'
import { cuid } from '@adonisjs/core/helpers'
import fs from 'fs/promises'
import path from 'path'

export default class NsfwDetectionService {
  private mlServiceUrl = 'http://localhost:8004'  // Back to original port

  /**
   * Analyze an uploaded file for NSFW content
   */
  async analyzeFile(file: MultipartFile): Promise<ModerationResult> {
    // Validate file
    if (!file.tmpPath) {
      throw new Error('Invalid file upload')
    }

    const fileType = this.getFileType(file.clientName || '')
    if (!['image', 'video'].includes(fileType)) {
      throw new Error('Only image and video files are supported')
    }

    // Save file to storage
    const fileName = `${cuid()}_${file.clientName}`
    const storagePath = app.makePath('storage/uploads')
    await fs.mkdir(storagePath, { recursive: true })

    // Move file to storage
    await file.move(storagePath, { name: fileName })
    const filePath = path.join(storagePath, fileName)

    // Create initial record
    const moderationResult = await ModerationResult.create({
      fileName: file.clientName || 'unknown',
      fileType: fileType as 'image' | 'video',
      filePath: filePath,
      isNsfw: false,
      nsfwScore: 0,
      detectionsCount: 0,
      status: 'pending'
    })

    try {
      const analysisResult = await this.callMlService(filePath, fileType)

      // Update record with results
      moderationResult.merge({
        isNsfw: analysisResult.is_nsfw,
        nsfwScore: analysisResult.nsfw_score,
        detectionsCount: analysisResult.detections_count || 0,
        bodyCoverage: analysisResult.body_coverage || null,
        clothingAnalysis: analysisResult.clothing_analysis || null,
        detailedDescription: analysisResult.detailed_description || null,
        coveragePercentage: analysisResult.coverage_percentage || null,
        detectedGender: analysisResult.detected_gender || null,
        confidenceScore: analysisResult.confidence_score || null,
        detailedAnalysis: analysisResult.detailed_analysis || null,
        analysisDetails: JSON.stringify({
          ...analysisResult,
          aspectsAnalysis: analysisResult.aspects_analysis || null
        }),
        status: 'completed'
      })

      await moderationResult.save()

      // CLEANUP: Delete the uploaded file after analysis
      try {
        await fs.unlink(filePath)
      } catch (cleanupError) {
        console.warn(`⚠️ Failed to cleanup file: ${filePath}`, cleanupError)
      }

      return moderationResult

    } catch (error) {
      // Update record with error
      moderationResult.merge({
        status: 'failed',
        errorMessage: error.message
      })
      await moderationResult.save()

      // CLEANUP: Delete the uploaded file even on error
      try {
        await fs.unlink(filePath)
      } catch (cleanupError) {
        console.warn(`⚠️ Failed to cleanup file after error: ${filePath}`, cleanupError)
      }

      throw error
    }
  }

  /**
   * Get all moderation results with pagination
   */
  async getResults(page: number = 1, limit: number = 20) {
    return await ModerationResult.query()
      .orderBy('created_at', 'desc')
      .paginate(page, limit)
  }

  /**
   * Get a specific moderation result
   */
  async getResult(id: number) {
    return await ModerationResult.findOrFail(id)
  }

  /**
   * Delete a moderation result and its file
   */
  async deleteResult(id: number) {
    const result = await ModerationResult.findOrFail(id)

    // Delete file if it exists
    try {
      await fs.unlink(result.filePath)
    } catch (error) {
      // File might not exist, continue with deletion
    }

    await result.delete()
  }

  /**
   * Call the ML service for analysis
   */
  private async callMlService(filePath: string, fileType: string) {
    const FormData = (await import('form-data')).default
    const fetch = (await import('node-fetch')).default

    const form = new FormData()
    const fileBuffer = await fs.readFile(filePath)
    form.append('file', fileBuffer, { filename: path.basename(filePath) })

    const endpoint = fileType === 'image' ? '/analyze-image' : '/analyze-video'
    const response = await fetch(`${this.mlServiceUrl}${endpoint}`, {
      method: 'POST',
      body: form
    })

    if (!response.ok) {
      throw new Error(`ML service error: ${response.statusText}`)
    }

    const result = await response.json() as any

    // Handle the NEW AI-powered response structure
    if (result.success && result.data) {
      return {
        is_nsfw: result.data.isNsfw,
        nsfw_score: result.data.detailedAnalysis?.nsfw_analysis?.nsfw_score || 0,
        detections_count: result.data.detailedAnalysis?.nsfw_analysis?.nsfw_detections?.length || 0,
        body_coverage: result.data.isNsfw ? 'NSFW Content Detected' : 'Safe Content',
        clothing_analysis: result.data.aiGeneratedDescription,
        detailed_description: result.data.aiGeneratedDescription,
        coverage_percentage: result.data.detailedAnalysis?.nsfw_analysis?.nsfw_score ?
          Math.round(result.data.detailedAnalysis.nsfw_analysis.nsfw_score * 100) : 0,
        detected_gender: result.data.detailedAnalysis?.face_analysis?.faces?.[0]?.gender || 'unknown',
        confidence_score: result.data.detailedAnalysis?.nsfw_analysis?.nsfw_score || 0,
        detailed_analysis: result.data.aiGeneratedDescription,
        aspects_analysis: {
          context: [result.data.detailedAnalysis?.object_analysis?.context || 'general'], // Make it an array
          activity: [result.data.detailedAnalysis?.pose_analysis?.activity_type || 'unknown'], // Make it an array
          faces_detected: result.data.facesDetected || 0,
          objects_detected: result.data.objectsDetected || 0,
          purpose: ['ai_analysis'],
          clothing: ['analysis_completed'],
          mood: ['neutral'],
          style: ['computer_vision_analysis']
        }
      }
    } else {
      throw new Error('Invalid response from ML service')
    }
  }

  /**
   * Determine file type from filename
   */
  private getFileType(filename: string): string {
    const ext = path.extname(filename).toLowerCase()
    const imageExts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    const videoExts = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm']

    if (imageExts.includes(ext)) return 'image'
    if (videoExts.includes(ext)) return 'video'
    return 'unknown'
  }
}