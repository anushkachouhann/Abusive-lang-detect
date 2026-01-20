import vine from '@vinejs/vine'

export const createLeoProfanityNoteValidator = vine.compile(
  vine.object({
    title: vine.string().trim().minLength(3).maxLength(200),
    content: vine.string().trim().minLength(10).maxLength(5000),
    language: vine.string().trim().optional(), // Optional language code for detection
  })
)

export const updateLeoProfanityNoteValidator = vine.compile(
  vine.object({
    title: vine.string().trim().minLength(3).maxLength(200).optional(),
    content: vine.string().trim().minLength(10).maxLength(5000).optional(),
    language: vine.string().trim().optional(), // Optional language code for detection
  })
)
