import { BaseSchema } from '@adonisjs/lucid/schema'

export default class extends BaseSchema {
  protected tableName = 'notes'

  async up() {
    this.schema.createTable(this.tableName, (table) => {
      table.increments('id')
      table.string('title').notNullable()
      table.text('content').notNullable()
      table.text('language').nullable()
      table.boolean('is_flagged').defaultTo(false)
      table.text('flagged_words').nullable()
      table.timestamp('created_at')
      table.timestamp('updated_at')
      table.boolean('is_checked_openai').notNullable().defaultTo(false)
    })
  }

  async down() {
    this.schema.dropTable(this.tableName)
  }
}