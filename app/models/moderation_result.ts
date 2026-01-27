import { DateTime } from 'luxon'
import { BaseModel, column } from '@adonisjs/lucid/orm'

export default class ModerationResult extends BaseModel {
  @column({ isPrimary: true })
  declare id: number

  @column({ columnName: 'file_name' })
  declare fileName: string

  @column({ columnName: 'file_type' })
  declare fileType: 'image' | 'video'

  @column({ columnName: 'file_path' })
  declare filePath: string

  @column({ columnName: 'is_nsfw' })
  declare isNsfw: boolean

  @column({ columnName: 'nsfw_score' })
  declare nsfwScore: number

  @column({ columnName: 'detections_count' })
  declare detectionsCount: number

  @column({ columnName: 'analysis_details' })
  declare analysisDetails: object | null

  @column({ columnName: 'body_coverage' })
  declare bodyCoverage: string | null

  @column({ columnName: 'clothing_analysis' })
  declare clothingAnalysis: string | null

  @column({ columnName: 'detailed_description' })
  declare detailedDescription: string | null

  @column({ columnName: 'coverage_percentage' })
  declare coveragePercentage: number | null

  @column({ columnName: 'detected_gender' })
  declare detectedGender: string | null

  @column({ columnName: 'confidence_score' })
  declare confidenceScore: number | null

  @column({ columnName: 'detailed_analysis' })
  declare detailedAnalysis: string | null

  @column()
  declare status: 'pending' | 'completed' | 'failed'

  @column({ columnName: 'error_message' })
  declare errorMessage: string | null

  @column.dateTime({ autoCreate: true })
  declare createdAt: DateTime

  @column.dateTime({ autoCreate: true, autoUpdate: true })
  declare updatedAt: DateTime
}