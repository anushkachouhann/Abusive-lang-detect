import User from '#models/user'
import { BaseSeeder } from '@adonisjs/lucid/seeders'
import hash from '@adonisjs/core/services/hash'

export default class extends BaseSeeder {
  async run() {
    await User.firstOrCreate(
      {
        fullName: 'Anu', 
        email: 'anu@gmail.com',
        password: await hash.make('anu1234')
      }
    ) 
  }
}
