import { BaseSchema } from '@adonisjs/lucid/schema'

export default class extends BaseSchema {
  protected tableName = 'moderation_results'

  async up() {
    this.schema.createTable(this.tableName, (table) => {
      table.increments('id')
      table.string('file_name').notNullable()
      table.string('file_type').notNullable()
      table.string('file_path').notNullable()
      table.boolean('is_nsfw').notNullable()
      table.decimal('nsfw_score', 5, 3).notNullable()
      table.integer('detections_count').defaultTo(0)
      table.json('analysis_details').nullable()
      table.string('status').defaultTo('completed')
      table.text('error_message').nullable()
      table.timestamp('created_at')
      table.timestamp('updated_at')
    })
  }

  async down() {
    this.schema.dropTable(this.tableName)
  }
}