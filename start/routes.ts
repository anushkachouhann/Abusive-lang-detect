/*
|--------------------------------------------------------------------------
| Routes file
|--------------------------------------------------------------------------
|
| The routes file is used for defining the HTTP routes.
|
*/

import router from '@adonisjs/core/services/router'
import { middleware } from './kernel.js'

const AuthController = () => import('#controllers/auth_controller') 
const LeoProfanityNotesController = () => import('#controllers/leo_profanity_notes_controller')

router.get('/', async () => {
  return {
    hello: 'world',
  }
}) 

// Auth routes (token-based)
router.post('/api/auth/register', [AuthController, 'register'])
router.post('/api/auth/login', [AuthController, 'login']) 
 
router
  .group(() => {
    router.get('/notes', [LeoProfanityNotesController, 'leoProfanityIndex'])
    router.post('/notes', [LeoProfanityNotesController, 'leoProfanityStore'])
    router.get('/notes/:id', [LeoProfanityNotesController, 'leoProfanityShow'])
    router.put('/notes/:id', [LeoProfanityNotesController, 'leoProfanityUpdate'])
    router.delete('/notes/:id', [LeoProfanityNotesController, 'leoProfanityDestroy'])
  })
  .prefix('/api/leo-profanity')
  .use(middleware.auth()) 