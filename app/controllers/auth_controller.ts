import type { HttpContext } from '@adonisjs/core/http'
import hash from '@adonisjs/core/services/hash'
import User from '#models/user'
import { loginValidator, registerValidator } from '#validators/auth'

export default class AuthController {
  async register({ request, response }: HttpContext) {
    const payload = await request.validateUsing(registerValidator)

    const existingUser = await User.findBy('email', payload.email)

    if (existingUser) {
      return response.conflict({
        success: false,
        message: 'Email is already registered',
      })
    }

    const user = await User.create({
      email: payload.email,
      password: payload.password,
      fullName: payload.fullName ?? null,
    })

    const token = await User.accessTokens.create(user, ['*'], { name: 'api-token' })
    const apiToken = token.value?.release()

    if (!apiToken) {
      return response.internalServerError({
        success: false,
        message: 'Failed to generate access token',
      })
    }

    return response.created({
      success: true,
      data: {
        user,
        token: apiToken,
      },
    })
  }

  async login({ request, response }: HttpContext) {
    const payload = await request.validateUsing(loginValidator)

    const user = await User.findBy('email', payload.email)

    if (!user) {
      return response.unauthorized({
        success: false,
        message: 'Invalid email or password',
      })
    }

    const token = await User.accessTokens.create(user, ['*'], { name: 'api-token' })
    const apiToken = token.value?.release()

    if (!apiToken) {
      return response.internalServerError({
        success: false,
        message: 'Failed to generate access token',
      })
    }

    return response.ok({
      success: true,
      data: {
        user,
        token: apiToken,
      },
    })
  }
}
