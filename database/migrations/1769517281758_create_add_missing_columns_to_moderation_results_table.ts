import { BaseSchema } from '@adonisjs/lucid/schema'

export default class extends BaseSchema {
  protected tableName = 'moderation_results'

  async up() {
    this.schema.alterTable(this.tableName, (table) => {
      table.string('body_coverage').nullable()
      table.text('clothing_analysis').nullable()
      table.text('detailed_description').nullable()
      table.decimal('coverage_percentage', 5, 2).nullable()
      table.string('detected_gender').nullable()
      table.decimal('confidence_score', 5, 3).nullable()
      table.text('detailed_analysis').nullable()
    })
  }

  async down() {
    this.schema.alterTable(this.tableName, (table) => {
      table.dropColumn('body_coverage')
      table.dropColumn('clothing_analysis')
      table.dropColumn('detailed_description')
      table.dropColumn('coverage_percentage')
      table.dropColumn('detected_gender')
      table.dropColumn('confidence_score')
      table.dropColumn('detailed_analysis')
    })
  }
}