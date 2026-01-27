/*
|--------------------------------------------------------------------------
| Routes file
|--------------------------------------------------------------------------
|
| The routes file is used for defining the HTTP routes.
|
*/

import router from '@adonisjs/core/services/router'
import app from '@adonisjs/core/services/app'

router.get('/', async () => {
  return {
    hello: 'world',
  }
})

// Serve test HTML file
router.get('/test', async ({ response }) => {
  return response.download(app.makePath('test.html'))
})

// NSFW Detection routes
router.group(() => {
  router.post('/analyze', '#controllers/moderations_controller.analyze')
  router.get('/results', '#controllers/moderations_controller.index')
  router.get('/results/:id', '#controllers/moderations_controller.show')
  router.delete('/results/:id', '#controllers/moderations_controller.destroy')
  router.get('/health', '#controllers/moderations_controller.health')
}).prefix('/api/moderation')
